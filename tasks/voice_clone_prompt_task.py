import asyncio
import logging
import uuid

import state
from database import SessionLocal
from models.voice_audio import VoiceAudio
from models.voice_clone_prompt import VoiceClonePrompt
from utils.voice_clone_prompt_utils import create_voice_clone_prompt

logger = logging.getLogger(__name__)


async def run_create_voice_clone_prompt_task(
    task_id: str,
    voice_audio_id: str,
    ref_wav_path,
    ref_audio_tokens_path,
    prompt_id,
    name,
    tags,
    ref_text,
):
    task = state.model_tasks[task_id]
    db = SessionLocal()

    async def add_log(message: str, type: str):
        entry = {
            "task_id": task_id,
            "message": message,
            "type": type,
            "prompt_id": prompt_id,
        }

        async with task["condition"]:
            task["logs"].append(entry)
            task["condition"].notify_all()

        logger.info("[%s] %s", task_id, message)

    try:
        await add_log("Start Process", "start")

        voice_audio = db.get(VoiceAudio, voice_audio_id)

        if voice_audio is None:
            raise ValueError("Voice audio not found")

        audio_length, ref_rms, ref_audio_tokens, ref_audio_tokens_path = (
            create_voice_clone_prompt(
                voice_audio,
                ref_wav_path,
                ref_audio_tokens_path,
            )
        )

        prompt = VoiceClonePrompt(
            id=prompt_id,
            name=name,
            tags=tags,
            ref_audio_tokens_path=str(ref_audio_tokens_path),
            ref_wav_path=str(ref_wav_path),
            ref_text=ref_text,
            ref_rms=ref_rms,
            voice_audio_id=voice_audio.id,
            audio_length=audio_length,
            ref_audio_tokens_shape=str(
                list(ref_audio_tokens.shape)
            ),
        )

        db.add(prompt)
        db.commit()
        db.refresh(prompt)

        await add_log("Finished Process", "finished")

    except Exception as exc:
        db.rollback()
        logger.exception("Task %s failed", task_id)
        await add_log(f"Process failed: {exc}", "error")

    finally:
        db.close()

        async with task["condition"]:
            task["done"] = True
            task["condition"].notify_all()


def start_create_voice_clone_prompt_task(
    voice_audio_id,
    ref_wav_path,
    ref_audio_tokens_path,
    prompt_id,
    name,
    tags,
    ref_text,
):
    task_id = str(uuid.uuid4())

    state.model_tasks[task_id] = {
        "logs": [],
        "name": "create_voice_clone_prompt",
        "done": False,
        "condition": asyncio.Condition(),
        "args": (
            voice_audio_id,
            ref_wav_path,
            ref_audio_tokens_path,
            prompt_id,
            name,
            tags,
            ref_text,
        ),
    }

    state.task_queue.put_nowait(task_id)

    return task_id