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
SAMPLE_RATE = 24_000