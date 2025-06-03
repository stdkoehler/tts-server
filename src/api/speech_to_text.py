"""Text to Speech endpoint"""

from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse
import tempfile
import whisper

from models.text_to_speech import TtsRequest
from src.core.dependencies import get_tts_models
from src.services.text_to_speech import TtsModelContainer


router = APIRouter(
    prefix="/transcribe",
    tags=["transcribe"],
    responses={404: {"description": "Not found"}},
)


@router.post("/speech-to-text", response_class=PlainTextResponse)
async def speech_to_text(file: UploadFile = File(...)) -> str:
    """
    Receives a webm file, transcribes it using Whisper, and returns the transcription as a string.
    """
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    # Load Whisper model (small for speed, change as needed)
    model = whisper.load_model("small")
    result = model.transcribe(tmp_path)
    return result["text"]
