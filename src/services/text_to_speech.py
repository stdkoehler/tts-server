"""Text to Speech Model"""

from dataclasses import dataclass
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydub import AudioSegment

import torch
import threading

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from f5_tts.api import F5TTS

from models.text_to_speech import TtsVoiceCoqui, TtsVoiceF5

from src.services.splitter import improved_split_sentence


# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
# https://www.vidavolta.io/streaming-with-fastapi/


class BaseModel:
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

        # Shadowrun terms
        text = text.replace("IC", "ice")
        text = text.replace("ICE", "ice")

        # Transform all-capital words into letters separated by points
        # (spell out the letters of abbreviations)
        # text = re.sub(r"\b([A-Z]{2,})\b", lambda m: ".".join(m.group(1)), text)

        chunks = [
            chunk.strip()
            for chunk in text.split("<break />")
            if any(c.isalpha() for c in chunk)
        ]
        return chunks


class CoquiModel(BaseModel):
    def __init__(self, output_path: Path, model_path: Path) -> None:
        self.ttsmodel: TtsVoiceCoqui | None = None
        self._gpt_cond_latent: torch.Tensor | None = None
        self._speaker_embedding: torch.Tensor | None = None

        self.output_path = output_path
        self.model_path = model_path
        # self.output_path = Path("output") / "xtts.wav"
        # self.model_path = Path("models")

        self._config = XttsConfig()
        self._config.load_json(self.model_path / "config.json")
        self._xtts_model = Xtts.init_from_config(self._config)

    def load_model(self, ttsmodel: TtsVoiceCoqui) -> None:
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


class F5Model(BaseModel):
    def __init__(self, output_path: Path, model_path: Path) -> None:
        self.output_path = output_path
        self.model_path = model_path
        self._ref_file: Path | None = None
        self._ref_text: str | None = None
        self._f5_model = F5TTS()

    def load_model(self, ttsmodel: TtsVoiceF5) -> None:
        self._ref_file = self.model_path / ttsmodel / "reference.wav"
        with open(
            self.model_path / ttsmodel / "reference.txt", "r", encoding="utf8"
        ) as f:
            reference_text = f.read()
        self._ref_text = reference_text

    def inference(self, text: str) -> None:

        chunks = self._preprocess_text_and_chunks(text)

        segments = []
        for chunk in chunks:
            out, _, _ = self._f5_model.infer(
                ref_file=self._ref_file,
                ref_text=self._ref_text,
                gen_text=chunk,
                file_wave=None,
                file_spec=None,
                seed=None,
            )

            wav = np.array(out)

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


@dataclass
class TtsModelContainer:
    coqui_model: CoquiModel
    f5_model: F5Model
