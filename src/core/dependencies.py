from fastapi import Request
from src.services.text_to_speech import TtsModelContainer


def get_tts_models(request: Request) -> TtsModelContainer:
    return request.app.state.tts_model_container
