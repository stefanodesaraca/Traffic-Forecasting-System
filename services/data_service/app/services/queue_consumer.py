import asyncio
from contextlib import suppress
from typing import Any

from shared_lib.rabbitmq import rabbitmq_client


async def handle_ingestion_message(payload: dict[str, Any]) -> None:
    message_id = payload.get("task_id")
    print(f"[data_service] received ingestion task: {message_id}")

    # Best-effort import to avoid import-time side effects during tests
    try:
        from services.data_service.app.db import update_ingestion_job
    except Exception:
        update_ingestion_job = None

    # Mark job as processing (best-effort, swallow DB/network errors)
    if update_ingestion_job:
        try:
            await update_ingestion_job(message_id, "processing")
        except Exception:
            pass

    # Perform processing with retries via a dedicated processing module
    try:
        from services.data_service.app.services.processing import (
            process_ingestion,
        )

        success = await process_ingestion(payload)

        if success:
            if update_ingestion_job:
                try:
                    await update_ingestion_job(message_id, "completed")
                except Exception:
                    pass
        else:
            if update_ingestion_job:
                try:
                    await update_ingestion_job(message_id, "failed")
                except Exception:
                    pass
    except Exception:
        # On unexpected failure, mark failed (best-effort)
        if update_ingestion_job:
            try:
                await update_ingestion_job(message_id, "failed")
            except Exception:
                pass


async def start_queue_consumer() -> None:
    await rabbitmq_client.consume("ingestion_queue", handle_ingestion_message)
    while True:
        await asyncio.sleep(1)


async def stop_queue_consumer() -> None:
    await rabbitmq_client.cancel_consumer()
    await rabbitmq_client.close()
