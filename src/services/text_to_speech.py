"""Text to Speech Model"""

import re
import subprocess
import textwrap
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydub import AudioSegment

import torch
import torchaudio
import threading

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import get_spacy_lang

from models.text_to_speech import TtsModel


# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
# https://www.vidavolta.io/streaming-with-fastapi/


def improved_split_sentence(text, lang, text_split_length=250):
    """
    Split text for TTS with nuanced chunk management.
    - Tries to keep chunks at or below text_split_length/2 (target)
    - Never appends NLP sentence or punctuation-split chunks past the target;
        start new chunk instead.
    - If a chunk must be forcibly split by textwrap (last resort), only allow
        artificial chunks to be appended together
        (past target, up to hard max) if the previous chunk was ALSO due to textwrap.

    By appending the sentences and punctuation splits only up to text_split_length/2
    leaves us slack to append textwrap chunks together and avoid unnatural split
    of sentences

    Args:
        text (str): The input text.
        lang (str): Language code for NLP processing.
        text_split_length (int): Max length of each split chunk.

    Returns:
        List[str]: The resulting split strings.
    """

    max_text_split_length = text_split_length
    target_text_split_length = int(max_text_split_length / 2)

    def split_on_punct(sentence, puncts=",;:"):
        parts = re.split(f"([{re.escape(puncts)}])", sentence)
        combined = []
        buf = ""
        for part in parts:
            buf += part
            if part in puncts:
                combined.append((buf.strip(), "punct"))
                buf = ""
        if buf.strip():
            combined.append((buf.strip(), "punct"))
        return combined

    # Maintain two parallel lists: one for text, one for chunk types
    text_splits = []
    split_types = []

    if len(text) >= target_text_split_length:
        nlp = get_spacy_lang(lang)
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        units = [str(sentence) for sentence in doc.sents]

        # Each element in pending_chunks is a tuple: (text, chunk_type)
        pending_chunks = []
        for unit in units:
            # If a sentence/unit is bigger than soft max, split further.
            if len(unit) > target_text_split_length:
                # Try punctuation split before hard wrapping
                punct_chunks = split_on_punct(unit)
                for chunk, _ in punct_chunks:
                    if len(chunk) > target_text_split_length:
                        # Really long, hard-wrap
                        for forced in textwrap.wrap(
                            chunk,
                            width=target_text_split_length,
                            drop_whitespace=True,
                            break_on_hyphens=False,
                            tabsize=1,
                        ):
                            pending_chunks.append((forced, "textwrap"))
                    else:
                        pending_chunks.append((chunk, "punct"))
            else:
                pending_chunks.append((unit, "nlp"))

        for chunk, chunk_type in pending_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if not text_splits:  # First chunk
                text_splits.append(chunk)
                split_types.append(chunk_type)
                continue

            current = text_splits[-1]
            current_type = split_types[-1]

            if len(current) < target_text_split_length:
                if (len(current) + 1 + len(chunk) <= target_text_split_length) or (
                    chunk_type == "textwrap"
                    and current_type == "textwrap"
                    and len(current) + 1 + len(chunk) <= max_text_split_length
                ):
                    text_splits[-1] = (current + " " + chunk).strip()
                    split_types[-1] = (
                        "textwrap"
                        if current_type == "textwrap" and chunk_type == "textwrap"
                        else current_type
                    )
                else:
                    text_splits.append(chunk)
                    split_types.append(chunk_type)
            else:
                # Only allow artificial (textwrap) chunks to be appended beyond target, up to hard max,
                # if BOTH current and incoming are textwrap
                if (
                    chunk_type == "textwrap"
                    and current_type == "textwrap"
                    and len(current) + 1 + len(chunk) <= max_text_split_length
                ):
                    text_splits[-1] = (current + " " + chunk).strip()
                    split_types[-1] = "textwrap"
                else:
                    text_splits.append(chunk)
                    split_types.append(chunk_type)
    else:
        text_splits = [text.strip()]

    text_splits = [chunk.rstrip('.,;:-)"') for chunk in text_splits]
    for text_split in text_splits:
        print(text_split)

    return text_splits


class Model:
    def __init__(self, output_path: Path, model_path: Path) -> None:
        self.ttsmodel: TtsModel | None = None
        self._gpt_cond_latent: torch.Tensor | None = None
        self._speaker_embedding: torch.Tensor | None = None

        self.output_path = output_path
        self.model_path = model_path
        # self.output_path = Path("output") / "xtts.wav"
        # self.model_path = Path("models")

        self._config = XttsConfig()
        self._config.load_json(self.model_path / "config.json")
        self._xtts_model = Xtts.init_from_config(self._config)

    def load_model(self, ttsmodel: TtsModel) -> None:
        if self.ttsmodel is not None and self.ttsmodel == ttsmodel:
            print("Model already loaded")
            return

        voice_path = self.model_path / ttsmodel

        self._xtts_model.load_checkpoint(
            self._config, checkpoint_dir=voice_path
        )  # , use_deepspeed=True)
        if torch.cuda.is_available():
            self._xtts_model.cuda()

        self._gpt_cond_latent, self._speaker_embedding = (
            self._xtts_model.get_conditioning_latents(
                audio_path=voice_path / "reference.wav",
            )
        )

        self.ttsmodel = ttsmodel
        print("Model loaded")

    def _preprocess_text_and_chunks(self, text: str):
        """
        Preprocess the input text and generate chunks for inference.
        Args:
            text (str): The input text.
        Returns:
            List[str]: List of processed text chunks.
        """
        text = text.replace("*", "").replace("---", "").replace("\\", "")
        text = text.replace("\n", "<break />").replace("...", "<break />")
        chunks = [
            chunk.strip()
            for chunk in text.split("<break />")
            if any(c.isalpha() for c in chunk)
        ]
        return chunks

    def inference(self, text: str) -> None:
        if self.ttsmodel is None:
            raise ValueError("Model not loaded")

        chunks = self._preprocess_text_and_chunks(text)

        segments = []
        with patch("TTS.tts.models.xtts.split_sentence", new=improved_split_sentence):
            for chunk in chunks:
                out = self._xtts_model.inference(
                    chunk,
                    "en",
                    self._gpt_cond_latent,
                    self._speaker_embedding,
                    enable_text_splitting=True,
                )

                wav = np.array(out["wav"])

                # Normalize to int16 for pydub
                wav_int16 = np.int16(wav / np.max(np.abs(wav)) * 32767)

                segment = AudioSegment(
                    wav_int16.tobytes(),
                    frame_rate=24000,
                    sample_width=2,  # int16
                    channels=1,
                )
                segments.append(segment)

        # 3. Add 1s of silence between segments
        pause = AudioSegment.silent(duration=700)  # pause

        combined_audio = segments[0] if segments else AudioSegment.silent(duration=100)
        for seg in segments[1:]:
            combined_audio += pause
            combined_audio += seg

        # 4. Export as MP3 and WAV
        combined_audio.export(
            self.output_path / "xtts.mp3", format="mp3", bitrate="320k"
        )

    def inference_generator_webm_opus(self, text: str):
        if self.ttsmodel is None:
            raise ValueError("Model not loaded")

        chunks = self._preprocess_text_and_chunks(text)

        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "-loglevel",
            "error",
            "pipe:1",
        ]

        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        def write_audio():
            try:
                for chunk in chunks:
                    print(chunk)
                    out = self._xtts_model.inference(
                        chunk,
                        "en",
                        self._gpt_cond_latent,
                        self._speaker_embedding,
                        enable_text_splitting=True,
                    )

                    wav = np.array(out["wav"])
                    wav_int16 = np.int16(wav / np.max(np.abs(wav)) * 32767)
                    ffmpeg_proc.stdin.write(wav_int16.tobytes())
            except Exception as e:
                print(f"Writer thread exception: {e}")
            finally:
                try:
                    ffmpeg_proc.stdin.close()
                except Exception:
                    pass

        writer_thread = threading.Thread(target=write_audio)
        writer_thread.start()

        try:
            while True:
                data = ffmpeg_proc.stdout.read(4096)
                if not data:
                    break
                yield data
        finally:
            ffmpeg_proc.kill()
            writer_thread.join()
