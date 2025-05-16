"""Text to Speech endpoint"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, StreamingResponse

from models.text_to_speech import TtsRequest
from src.core.dependencies import get_tts_models
from src.services.text_to_speech import TtsModelContainer


router = APIRouter(
    prefix="/inference",
    tags=["inference"],
    responses={404: {"description": "Not found"}},
)


@router.post("/text-to-speech", response_class=FileResponse)
async def text_to_speech(
    prompt: TtsRequest, tts_model_container: TtsModelContainer = Depends(get_tts_models)
) -> FileResponse:
    # Return the audio file
    if prompt.model == "coqui":
        tts_model_container.coqui_model.load_model(prompt.voice)
        tts_model_container.coqui_model.inference(prompt.text)
        audio_file_path = tts_model_container.coqui_model.output_path / "xtts.mp3"
    else:
        tts_model_container.f5_model.load_model(prompt.voice)
        tts_model_container.f5_model.inference(prompt.text)
        audio_file_path = tts_model_container.f5_model.output_path / "xtts.mp3"

    print(audio_file_path)
    return FileResponse(
        audio_file_path,
        media_type="audio/mpeg",  # "audio/wav",  # or 'audio/mpeg' for mp3
        headers={"Content-Disposition": "attachment; filename=inference.mp3"},
    )


@router.post("/text-to-speech-stream-webm")
async def text_to_speech_stream_webm(
    prompt: TtsRequest, tts_model_container: TtsModelContainer = Depends(get_tts_models)
):
    """
    Streams synthesized Opus-in-WebM audio back to the client.
    Content-Type is audio/webm;codecs=opus.
    """
    tts_model_container.coqui_model.load_model(prompt.model)
    audio_gen = tts_model_container.coqui_model.inference_generator_webm_opus(
        prompt.text
    )
    return StreamingResponse(
        audio_gen,
        media_type="audio/webm;codecs=opus",
        headers={"Content-Disposition": "inline; filename=inference_stream.webm"},
    )
