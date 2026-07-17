import unittest
from unittest.mock import AsyncMock, patch
import importlib
import sys
import types

from services.api_gateway.app.routers.orchestration import run_ingestion


class TestConsumerFailurePath(unittest.IsolatedAsyncioTestCase):
    async def test_consumer_marks_failed_on_processing_exception(self):
        # stub asyncpg to avoid import-time errors
        asyncpg_stub = types.ModuleType("asyncpg")

        async def _connect(*args, **kwargs):
            raise RuntimeError("asyncpg not available in test environment")

        asyncpg_stub.connect = _connect
        sys.modules.setdefault("asyncpg", asyncpg_stub)

        # make process_ingestion raise
        process_mock = AsyncMock(side_effect=Exception("processing failed"))

        # prepare update_ingestion_job mock
        update_mock = AsyncMock()

        async def fake_publish(queue_name: str, payload: dict):
            from services.data_service.app.services.queue_consumer import (
                handle_ingestion_message,
            )

            await handle_ingestion_message(payload)

        db_module = importlib.import_module("services.data_service.app.db")

        with patch(
            "services.api_gateway.app.routers.orchestration.rabbitmq_client.publish",
            new=fake_publish,
        ), patch(
            "services.data_service.app.services.processing.process_ingestion",
            new=process_mock,
        ), patch.object(db_module, "update_ingestion_job", update_mock):
            result = await run_ingestion("fail-project")

        self.assertEqual(result["status"], "queued")

        # update_ingestion_job should be called to set 'processing' then 'failed'
        calls = [call.args[1] for call in update_mock.await_args_list]
        self.assertIn("processing", calls)
        self.assertIn("failed", calls)


if __name__ == "__main__":
    unittest.main()
