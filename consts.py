import os
from dotenv import load_dotenv

load_dotenv()

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".opus",
    ".webm",
}

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOAD_ASR = os.getenv("LOAD_ASR", "false").lower() == "true"