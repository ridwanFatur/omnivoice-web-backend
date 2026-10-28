import asyncio
import logging

import state
from consts import DEBUG
from tasks.dummy_task import run_dummy_task
from tasks.voice_clone_prompt_task import run_create_voice_clone_prompt_task
from tasks.voice_generation_task import run_generate_voice_task

logger = logging.getLogger(__name__)

TASK_RUNNERS = {
    "dummy_process": run_dummy_task,
    "create_voice_clone_prompt": run_create_voice_clone_prompt_task,
    "generate_voice": run_generate_voice_task,
}


async def task_worker():
    while True:
        task_id = await state.task_queue.get()

        try:
            task = state.model_tasks[task_id]
            name = task["name"]
            runner = TASK_RUNNERS[name]

            if DEBUG and name in {
                "create_voice_clone_prompt",
                "generate_voice",
            }:
                await asyncio.sleep(10)

            await runner(task_id, *task.get("args", ()))

        except Exception:
            logger.exception("Task %s failed", task_id)

        finally:
            state.task_queue.task_done()