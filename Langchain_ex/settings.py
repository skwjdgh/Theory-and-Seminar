import os
from pydantic import BaseSettings
import torch

class Settings(BaseSettings):
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    EMBEDDING_MODEL: str = "llama3.1"
    WHISPER_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        env_file = ".env"
