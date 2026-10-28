import asyncio

model = None
model_tasks = {}
task_queue = asyncio.Queue()