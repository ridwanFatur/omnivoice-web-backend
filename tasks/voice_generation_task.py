import asyncio
import logging
import uuid
import time
from dataclasses import asdict

import numpy as np
import soundfile as sf

import state
from database import SessionLocal
from models.voice_clone_prompt import VoiceClonePrompt
from models.voice_generation import VoiceGeneration
from utils.generation_config import GenerationConfig, GenerationParameter
from utils.voice_generation_utils import generate_voice

logger = logging.getLogger(__name__)

async def run_generate_voice_task(
    task_id: str,
    text: str,
    generation_id: str,
    voice_clone_prompt_id,
    generation_parameter,
    generation_config,
    audio_path,
    name,
):
    task = state.model_tasks[task_id]
    db = SessionLocal()

    async def add_log(message: str, type: str):
        entry = {
            "task_id": task_id,
            "message": message,
            "type": type,
            "generation_id": generation_id,
        }

        async with task["condition"]:
            task["logs"].append(entry)
            task["condition"].notify_all()

        logger.info("[%s] %s", task_id, message)

    try:
        await add_log("Start Process", "start")

        voice_clone_prompt = db.get(
            VoiceClonePrompt,
            voice_clone_prompt_id,
        )

        if voice_clone_prompt is None:
            raise ValueError("Voice clone prompt not found")

        generation_parameter = GenerationParameter.from_dict(
            generation_parameter,
        )
        generation_config = GenerationConfig.from_dict(
            generation_config,
        )

        combined_config = {
            "generation_config": asdict(generation_config),
            "generation_parameter": asdict(generation_parameter),
        }

        generated_audios, sample_rate = generate_voice(
            text=text,
            voice_clone_prompt=voice_clone_prompt,
            generation_config=generation_config,
            generation_parameter=generation_parameter,
        )

        audio_lengths = []

        for index, generated_audio in enumerate(generated_audios):
            generated_audio = np.asarray(
                generated_audio,
                dtype=np.float32,
            ).squeeze()

            if generated_audio.ndim != 1:
                raise ValueError(
                    f"Generated audio at index {index} must be a mono waveform"
                )

            audio_length = round(len(generated_audio) / sample_rate)
            audio_lengths.append(audio_length)

            output_path = audio_path.with_name(
                f"{audio_path.name}-{index}.wav"
            )

            sf.write(
                str(output_path),
                generated_audio,
                sample_rate,
                format="WAV",
            )

        generation = VoiceGeneration(
            id=generation_id,
            text=text,
            created_at=int(time.time() * 1000),
            name=name,
            path=str(audio_path),
            generated_count=len(generated_audios),
            audio_lengths=audio_lengths,
            voice_clone_prompt_id=voice_clone_prompt.id,
            generation_config=combined_config,
        )

        db.add(generation)
        db.commit()
        db.refresh(generation)

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

def start_generate_voice_task(
    text,
    generation_id,
    voice_clone_prompt_id,
    generation_parameter,
    generation_config,
    audio_path,
    name,
):
    task_id = str(uuid.uuid4())

    state.model_tasks[task_id] = {
        "logs": [],
        "name": "generate_voice",
        "done": False,
        "condition": asyncio.Condition(),
        "args": (
            text,
            generation_id,
            voice_clone_prompt_id,
            generation_parameter,
            generation_config,
            audio_path,
            name,
        ),
    }

    state.task_queue.put_nowait(task_id)

    return task_id