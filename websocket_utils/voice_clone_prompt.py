import uuid
from consts import DEBUG
from models.voice_clone_prompt import VoiceClonePrompt
from omnivoice_utils.prepare_omnivoice import model_tasks
import asyncio
from database import SessionLocal
from omnivoice_utils.voice_clone_prompt_utils import create_voice_clone_prompt
from models.voice_audio import VoiceAudio
import logging

logger = logging.getLogger(__name__)

async def run_create_voice_clone_prompt(
    task_id: str,
    voice_audio_id: str,
    ref_wav_path,
    ref_audio_tokens_path,
    prompt_id,
    name,
    tags,
    ref_text,
):
    task = model_tasks[task_id]
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
        
    await add_log("Start Process", "start")
    try:
        voice_audio = db.get(VoiceAudio, voice_audio_id)

        if voice_audio is None:
            raise ValueError("Voice audio not found")

        audio_length, ref_rms, ref_audio_tokens = (
            create_voice_clone_prompt(
                voice_audio,
                ref_wav_path,
                ref_audio_tokens_path,
            )
        )

        if DEBUG:
            await asyncio.sleep(10)

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
        await add_log(f"Process failed: {exc}", "error")
    finally:
        db.close()  
        async with task["condition"]:
            task["done"] = True
            task["condition"].notify_all()
    

def start_create_voice_clone_prompt(
    voice_audio_id, 
    ref_wav_path, 
    ref_audio_tokens_path,
    prompt_id,
    name,
    tags,
    ref_text,
):
    task_id = str(uuid.uuid4())
    model_tasks[task_id] = {
        "logs": [],
        "name": "create_voice_clone_prompt",
        "done": False,
        "condition": asyncio.Condition(),
    }
    asyncio.create_task(
        run_create_voice_clone_prompt(
            task_id,
            voice_audio_id, 
            ref_wav_path, 
            ref_audio_tokens_path,
            prompt_id,
			name,
			tags,
			ref_text,
        )
    )
    return task_id