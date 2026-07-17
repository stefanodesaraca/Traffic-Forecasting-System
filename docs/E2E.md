End-to-end (E2E) test instructions

This project includes a lightweight Docker Compose file for running RabbitMQ and Postgres locally for E2E tests.

Prerequisites

- Docker & Docker Compose installed and running on your machine.
- Python dev dependencies installed (see `requirements.txt`).

Run E2E tests (PowerShell)

```powershell
cd <repo-root>
# Start docker services and run E2E test
scripts\run_e2e.ps1
```

Notes

- The E2E test uses the local images defined in `docker-compose.e2e.yml` and runs the test `services/integration_tests/test_e2e_compose.py`.
- The test will only run if `ENABLE_E2E=1` is set by the runner script; avoid running it as part of fast unit-test suites.
- The runner sets `POSTGRES_HOST` and `RABBITMQ_HOST` to `localhost` so the host python process can reach the containers on forwarded ports.
- Running this on CI: ensure the CI runner has Docker and sufficient permissions. Alternatively run the E2E tests inside a dedicated container that shares the compose network.
