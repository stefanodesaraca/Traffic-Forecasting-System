import unittest
from unittest.mock import AsyncMock, patch

from services.api_gateway.app.routers.orchestration import run_ingestion


class TestGatewayToDataIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_publishes_and_consumer_processes(self):
        # Prepare a publish mock that will call the consumer handler
        async def fake_publish(queue_name: str, payload: dict):
            # import the consumer handler and call it as the broker would
            from services.data_service.app.services.queue_consumer import (
                handle_ingestion_message,
            )

            await handle_ingestion_message(payload)

        update_mock = AsyncMock()

        import importlib
        import sys
        import types

        # Prevent import-time dependency on asyncpg during tests
        asyncpg_stub = types.ModuleType("asyncpg")

        async def _connect(*args, **kwargs):
            raise RuntimeError("asyncpg not available in test environment")

        asyncpg_stub.connect = _connect
        sys.modules.setdefault("asyncpg", asyncpg_stub)

        db_module = importlib.import_module("services.data_service.app.db")

        with patch(
            "services.api_gateway.app.routers.orchestration.rabbitmq_client.publish",
            new=fake_publish,
        ), patch.object(db_module, "update_ingestion_job", update_mock):
            result = await run_ingestion("integration-project")

        # Gateway should return queued status
        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["project_id"], "integration-project")

        # Consumer should have attempted to mark job processing then completed
        # We expect at least one call to update_ingestion_job
        self.assertTrue(update_mock.await_count >= 1)


if __name__ == "__main__":
    unittest.main()
