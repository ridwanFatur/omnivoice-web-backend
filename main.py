from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import logging
from websocket_utils.dummy_process import start_dummy_process
from omnivoice_utils.prepare_omnivoice import load_model
from routers.voice_audio import router as voice_audio_router
from routers.voice_clone_prompt import router as voice_clone_prompt_router
from routers.voice_generation import router as voice_generation_router
from database import engine
from routers.web_socket import (
    router as web_socket_router,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("FastAPI startup")
    load_model()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_audio_router)
app.include_router(voice_clone_prompt_router)
app.include_router(voice_generation_router)
app.include_router(web_socket_router)

BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/audio-files",
    StaticFiles(directory=BASE_DIR / "audio-files"),
    name="audio-files",
)

@app.get("/health")
def health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "ok",
            "database": "connected",
        }
    except Exception:
        return {
            "status": "error",
            "database": "disconnected",
        }
        
@app.post("/test")
async def trigger_test_process():
    task_id = start_dummy_process()

    return {
        "task_id": task_id,
        "status": "started",
        "websocket_url": f"/ws/logs/{task_id}",
    }