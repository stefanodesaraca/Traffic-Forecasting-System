# Changelog

## [Unreleased] - 2026-07-17

### Added
- Data service: improved queue consumer processing with retry/backoff and best-effort DB status updates (`services/data_service/app/services/processing.py`, `services/data_service/app/services/queue_consumer.py`).
- Integration test simulating gateway -> RabbitMQ -> consumer flow (`services/integration_tests/test_gateway_to_data_via_rabbitmq.py`).

### Changed
- API Gateway: orchestration publishes to RabbitMQ with HTTP fallback; added resilience to publish failures (`services/api_gateway/app/routers/orchestration.py`).

### Tests
- Unit tests for `data_service` and `api_gateway` pass locally.
- Integration test added and passing (simulated RabbitMQ publish).

### Notes
- Consumer processing logic is currently a placeholder and should be replaced with the real `src.pipelines` ingestion calls in a follow-up.
- For full end-to-end verification, consider running a RabbitMQ test container and a test Postgres instance in CI.
