from fastapi import Request
from src.services.text_to_speech import Model


def get_tts_model(request: Request) -> Model:
    return request.app.state.tts_model
