import os
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from services.api_gateway.app.routers.orchestration import run_ingestion


class TestE2ECompose(IsolatedAsyncioTestCase):
    async def test_end_to_end_compose(self):
        if os.getenv("ENABLE_E2E") != "1":
            self.skipTest("Enable E2E by setting ENABLE_E2E=1 and starting docker-compose.e2e.yml")

        # Allow services some time to settle
        await asyncio_sleep(0.5)

        result = await run_ingestion("e2e-project")
        self.assertEqual(result["status"], "queued")

        # Poll Postgres until job appears (timeout)
        timeout = 30
        interval = 1
        from shared_lib.database import postgres_connection

        found = False
        end = time.time() + timeout
        while time.time() < end:
            try:
                async with postgres_connection() as conn:
                    row = await conn.fetchrow(
                        "SELECT * FROM ingestion_jobs WHERE project_id = $1 LIMIT 1",
                        "e2e-project",
                    )
                    if row:
                        found = True
                        break
            except Exception:
                pass
            await asyncio_sleep(interval)

        self.assertTrue(found, "Ingestion job not found in Postgres within timeout")


async def asyncio_sleep(s):
    import asyncio

    await asyncio.sleep(s)


if __name__ == "__main__":
    unittest.main()
