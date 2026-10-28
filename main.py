from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import logging
from database import engine
from routers import voice_audio_router, voice_clone_prompt_router, voice_generation_router, web_socket_task_router
from tasks.dummy_task import start_dummy_task
from tasks.task_worker import task_worker
from utils.model_loader import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from utils.model_loader import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("FastAPI startup")

    load_model()

    worker = asyncio.create_task(task_worker())

    try:
        yield
    finally:
        worker.cancel()

        try:
            await worker
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_audio_router.router)
app.include_router(voice_clone_prompt_router.router)
app.include_router(voice_generation_router.router)
app.include_router(web_socket_task_router.router)

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
    task_id = start_dummy_task()

    return {
        "task_id": task_id,
        "status": "started",
        "websocket_url": f"/ws/logs/{task_id}",
    }