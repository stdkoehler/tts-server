"""FastAPI Server"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.text_to_speech import router as speech_router
from src.services.text_to_speech import Model
from models.text_to_speech import TtsModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path("models")
    output_path = Path("output")

    # Initialize and load the TTS model
    tts_model = Model(output_path, model_path)
    tts_model.load_model(TtsModel.Callum)  # Load default voice

    # Store the model in the application state
    app.state.tts_model = tts_model
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
