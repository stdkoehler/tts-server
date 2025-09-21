"""FastAPI Server"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.text_to_speech import router as speech_router
from src.api.speech_to_text import router as stt_router
from src.services.text_to_speech import (
    TtsModelContainer,
    CoquiModel,
    F5Model,
    VoxCpmModel,
)
from models.text_to_speech import TtsVoiceCoqui


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path("models")
    output_path = Path("output")

    # Initialize and load the TTS model
    tts_coqui_model = CoquiModel(
        output_path=output_path, model_path=model_path / "coqui"
    )
    tts_coqui_model.load_model(TtsVoiceCoqui.Callum)  # Load default voice

    tts_f5_model = F5Model(output_path=output_path, model_path=model_path / "f5")

    tts_vox_cpm_model = VoxCpmModel(
        output_path=output_path, model_path=model_path / "f5"
    )

    # Store the model in the application state
    app.state.tts_model_container = TtsModelContainer(
        coqui_model=tts_coqui_model,
        f5_model=tts_f5_model,
        vox_cpm_model=tts_vox_cpm_model,
    )

    print("TTS model loaded and ready.")

    yield

    # Optional: Add any cleanup logic here
    print("Shutting down application.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(speech_router)
app.include_router(stt_router)
