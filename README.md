# TTS Server

## Todo
- https://huggingface.co/spaces/hexgrad/Kokoro-TTS
- https://huggingface.co/spaces/ResembleAI/Chatterbox

## Overview
This project is a Text-to-Speech (TTS) server built using FastAPI and coqui. It provides endpoints for generating speech from text using pre-trained TTS models. The server supports both file-based and streaming audio responses.

## Features
- **Text-to-Speech Conversion**: Generate audio files or stream audio directly from text input.
- **Multiple Voices**: Supports multiple pre-trained TTS models.
- **Custom Sentence Splitting**: Implements an improved sentence splitting algorithm for better TTS chunking.
- **Streaming Support**: Streams audio in WebM format with Opus codec.

## Requirements
- Python 3.12
- CUDA-enabled GPU (for faster inference)
- FFmpeg (for audio processing)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd tts-server
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Install FFmpeg (if not already installed):
   - On Ubuntu:
     ```bash
     sudo apt install ffmpeg
     ```
   - On Windows:
     Download and install FFmpeg from [https://ffmpeg.org/](https://ffmpeg.org/).

## Usage

### Running the Server
Start the FastAPI server:
```bash
poetry run python run.py
```
The server will be available at `http://0.0.0.0:8001`.

### Endpoints
#### 1. `/inference/text-to-speech`
- **Method**: POST
- **Description**: Converts text to speech and returns an audio file.
- **Request Body**:
  ```json
  {
    "text": "Your input text here",
    "model": "Model"
  }
  ```
- **Response**: MP3 file.

#### 2. `/inference/text-to-speech-stream-webm`
- **Method**: POST
- **Description**: Streams synthesized audio in WebM format.
- **Request Body**:
  ```json
  {
    "text": "Your input text here",
    "model": "Model"
  }
  ```
- **Response**: WebM audio stream.

## Models
The server supports multiple pre-trained coqui TTS models stored in the `models/` directory. Each model directory should contain:
- `config.json`
- `model.pth`
- `reference.wav`
- `speakers_xtts.pth`
- `vocab.json`

## Development

### Testing
Run tests using pytest:
```bash
poetry run pytest
```

### Debugging
Use the provided VS Code launch configurations for debugging:
- **Python Debugger: Run**: Runs the server.
- **Python Debugger: Current File**: Debugs the currently open file.

### Code Style
- The project uses `pylint` and `mypy` for linting and type checking.
- Format code on save in VS Code (`editor.formatOnSave` is enabled).

## License
This project is licensed under the MIT License.
