"""Text to Speech Model"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import subprocess
from pathlib import Path
from unittest.mock import patch
import contextlib

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


class BaseModel(ABC):
    def __init__(self, output_path: Path, model_path: Path) -> None:
        self.output_path = output_path
        self.model_path = model_path

    def _preprocess_text_and_chunks(self, text: str):
        """
        Preprocess the input text and generate chunks for inference.
        Args:
            text (str): The input text.
        Returns:
            List[str]: List of processed text chunks.
        """
        text = text.replace("*", "").replace("---", "").replace("\\", "")
        text = (
            text.replace("\n", "<break />")
            .replace("...", "<break />")
            .replace(":", "...")  # tts models prefer comma for pauses
            .replace("…", "...")
        )

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

    def _wav_to_segment(self, wav: np.ndarray) -> AudioSegment:
        """
        Convert a numpy wav array to a normalized int16 AudioSegment.
        """
        wav_int16 = np.int16(wav / np.max(np.abs(wav)) * 32767)
        segment = AudioSegment(
            wav_int16.tobytes(),
            frame_rate=24000,
            sample_width=2,  # int16
            channels=1,
        )
        return segment

    def _combine_and_export_segments(
        self,
        segments: list[AudioSegment],
        output_path: Path,
        filename: str = "xtts.mp3",
    ):
        """
        Combine audio segments with 700ms silence and export as MP3.
        """
        pause = AudioSegment.silent(duration=700)  # pause
        combined_audio = segments[0] if segments else AudioSegment.silent(duration=100)
        for seg in segments[1:]:
            combined_audio += pause
            combined_audio += seg
        combined_audio.export(output_path / filename, format="mp3", bitrate="320k")

    def _stream_webm_opus(self, chunks):
        """
        Shared streaming logic for webm/opus output. chunk_to_wav_fn(chunk) -> np.ndarray
        """
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
                    wav = self._chunk_to_wav(chunk)
                    wav_int16 = np.int16(wav / np.max(np.abs(wav)) * 32767)
                    ffmpeg_proc.stdin.write(wav_int16.tobytes())
                    # Add 700ms pause (silence) after each chunk
                    pause_samples = int(0.7 * 24000)  # 700ms at 24kHz
                    silence = np.zeros(pause_samples, dtype=np.int16)
                    ffmpeg_proc.stdin.write(silence.tobytes())
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

    @abstractmethod
    def _chunk_to_wav(self, chunk):
        """
        Abstract method to convert a chunk to a wav numpy array.
        """
        pass

    def inference_generator_webm_opus(self, text: str):
        chunks = self._preprocess_text_and_chunks(text)
        yield from self._stream_webm_opus(chunks)

    def _inference_context(self):
        """
        Context manager for model-specific inference context. Override in subclasses if needed.
        """
        return contextlib.nullcontext()

    def inference(self, text: str) -> None:
        chunks = self._preprocess_text_and_chunks(text)
        segments = []
        with self._inference_context():
            for chunk in chunks:
                wav = self._chunk_to_wav(chunk)
                segment = self._wav_to_segment(wav)
                segments.append(segment)
        self._combine_and_export_segments(segments, self.output_path)


class CoquiModel(BaseModel):
    def __init__(self, output_path: Path, model_path: Path) -> None:
        super().__init__(output_path, model_path)
        self.ttsmodel: TtsVoiceCoqui | None = None
        self._gpt_cond_latent: torch.Tensor | None = None
        self._speaker_embedding: torch.Tensor | None = None

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

    def _inference_context(self):
        """
        Use a patch context manager to override the sentence splitter in Coqui TTS.
        This ensures improved sentence splitting for more natural speech synthesis.
        """
        return patch("TTS.tts.models.xtts.split_sentence", new=improved_split_sentence)

    def _chunk_to_wav(self, chunk):
        out = self._xtts_model.inference(
            chunk,
            "en",
            self._gpt_cond_latent,
            self._speaker_embedding,
            enable_text_splitting=True,
        )
        return np.array(out["wav"])


class F5Model(BaseModel):
    def __init__(self, output_path: Path, model_path: Path) -> None:
        super().__init__(output_path, model_path)
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

    def _inference_context(self):
        """
        Use torch.no_grad() to disable gradient calculation during inference.
        This reduces memory usage and speeds up computations, as gradients are not needed for TTS inference.
        """
        return torch.no_grad()

    def _chunk_to_wav(self, chunk):
        out, _, _ = self._f5_model.infer(
            ref_file=self._ref_file,
            ref_text=self._ref_text,
            gen_text=chunk,
            file_wave=None,
            file_spec=None,
            seed=None,
        )
        return np.array(out)


@dataclass
class TtsModelContainer:
    coqui_model: CoquiModel
    f5_model: F5Model
