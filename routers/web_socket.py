import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from omnivoice_utils.prepare_omnivoice import model_tasks

router = APIRouter(tags=["WebSocket"])
logger = logging.getLogger(__name__)

@router.websocket("/ws/logs/{task_id}")
async def process_logs(websocket: WebSocket, task_id: str):
    task = model_tasks.get(task_id)

    if task is None:
        await websocket.close(code=1008, reason="Task not found")
        return

    await websocket.accept()
    logger.info("WebSocket connected for task %s", task_id)

    index = 0

    try:
        while True:
            async with task["condition"]:
                await task["condition"].wait_for(
                    lambda: index < len(task["logs"]) or task["done"]
                )

                new_logs = task["logs"][index:]
                index += len(new_logs)
                done = task["done"]

            for log in new_logs:
                await websocket.send_json(log)

            if done and index >= len(task["logs"]):
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for task %s", task_id)
       
        

