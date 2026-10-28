import asyncio
import random
import uuid
import logging
import state

logger = logging.getLogger(__name__)

MESSAGES = [
    "Preparing voice model...",
    "Processing audio frames...",
    "Extracting voice features...",
    "Generating speech...",
    "Applying voice characteristics...",
]

async def run_dummy_task(task_id: str):
    task = state.model_tasks[task_id]

    async def add_log(message: str):
        entry = {
            "task_id": task_id,
            "message": message,
            "type": "dummy_process",
        }

        async with task["condition"]:
            task["logs"].append(entry)
            task["condition"].notify_all()

        logger.info("[%s] %s", task_id, message)

    try:
        await add_log("Process started")

        for i in range(1, 11):
            await asyncio.sleep(4)
            await add_log(
                f"Step {i}/10: {random.choice(MESSAGES)}"
            )

        await add_log("Process completed")

    except Exception as exc:
        await add_log(f"Process failed: {exc}")

    finally:
        async with task["condition"]:
            task["done"] = True
            task["condition"].notify_all()

def start_dummy_task() -> str:
    task_id = str(uuid.uuid4())

    state.model_tasks[task_id] = {
        "logs": [],
        "name": "dummy_process",
        "done": False,
        "condition": asyncio.Condition(),
    }

    asyncio.create_task(run_dummy_task(task_id))

    return task_id