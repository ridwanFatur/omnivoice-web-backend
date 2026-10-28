import logging
import uuid
from consts import DEBUG
from models.voice_clone_prompt import VoiceClonePrompt
from omnivoice_utils.prepare_omnivoice import model_tasks
import asyncio
import numpy as np
import soundfile as sf
from database import SessionLocal
import time
from omnivoice_utils.generate_audio_utils import generate_voice
from consts import SAMPLE_RATE
from omnivoice_utils.data_models import GenerationConfig, GenerationParameter
from dataclasses import asdict
from models.voice_generation import VoiceGeneration

logger = logging.getLogger(__name__)

async def run_generate_voice(
	task_id: str,
 	text: str,
 	generation_id: str,
  	voice_clone_prompt_id,
   	generation_parameter,
   	generation_config,
    audio_path,
    name
):
    task = model_tasks[task_id]
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
    await add_log("Start Process", "start")
    
    try:     
        voice_clone_prompt = db.get(VoiceClonePrompt, voice_clone_prompt_id)
        if voice_clone_prompt is None:
            raise ValueError("Voice clone prompt not found")
        
        if DEBUG:
            await asyncio.sleep(10)
            
        generation_parameter = GenerationParameter.from_dict(
			generation_parameter
		)
        generation_config = GenerationConfig.from_dict(
			generation_config
		)
        combined_config = {
			"generation_config": asdict(generation_config),
			"generation_parameter": asdict(generation_parameter),
		}
        
        generated_audio = generate_voice(
            text=text,
			voice_clone_prompt=voice_clone_prompt,
			generation_config=generation_config,
			generation_parameter=generation_parameter,
		)
        generated_audio = np.asarray(
			generated_audio,
			dtype=np.float32,
		).squeeze()
        
        if generated_audio.ndim != 1:
            raise ValueError("Generated audio must be a mono waveform")
        
        audio_length = round(len(generated_audio) / SAMPLE_RATE)
        
        sf.write(
            str(audio_path),
            generated_audio,
            SAMPLE_RATE,
            format="WAV",
        )
        generation = VoiceGeneration(
            id=generation_id,
            text=text,
            created_at=int(time.time() * 1000),
            name=name,
            path=str(audio_path),
            audio_length=audio_length,
            voice_clone_prompt_id=(
                voice_clone_prompt.id
                if voice_clone_prompt is not None
                else None
            ),
            generation_config=combined_config,
        )

        db.add(generation)
        db.commit()
        db.refresh(generation)
        await add_log("Finished Process", "finished")
        
    except Exception as exc:
        db.rollback()
        await add_log(f"Process failed: {exc}", "error")
    finally:
        db.close()  
        async with task["condition"]:
            task["done"] = True
            task["condition"].notify_all()       


def start_generate_voice(
    text,
	generation_id,
 	voice_clone_prompt_id,
   	generation_parameter,
   	generation_config,
    audio_path,
    name
):
    task_id = str(uuid.uuid4())
    model_tasks[task_id] = {
        "logs": [],
        "name": "generate_voice",
        "done": False,
        "condition": asyncio.Condition(),
    }    
    asyncio.create_task(
		run_generate_voice(
			task_id,
   			text,
   			generation_id,
      		voice_clone_prompt_id,
			generation_parameter,
			generation_config,
   			audio_path,
      		name
		)
	)
    return task_id
