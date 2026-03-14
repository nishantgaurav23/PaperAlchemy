# S9b.7 — Docker Platform Services — Checklist

## Phase 1: Tests (Red)
- [x] Write test_compose_platform_services
- [x] Write test_minio_service_config
- [x] Write test_sandbox_service_config
- [x] Write test_api_env_vars
- [x] Write test_env_example_minio_vars
- [x] Write test_env_example_sandbox_vars
- [x] Write test_makefile_platform_target
- [x] Write test_volumes_declared
- [x] All tests fail (RED confirmed)

## Phase 2: Implementation (Green)
- [x] Add MinIO service to compose.yml
- [x] Add sandbox (DinD) service to compose.yml
- [x] Add minio_data and sandbox_data volumes
- [x] Update API service environment with new vars
- [x] Add MINIO__* and SANDBOX__* vars to .env.example
- [x] Add `platform` target to Makefile
- [x] All tests pass (GREEN confirmed)

## Phase 3: Refactor & Verify
- [x] `docker compose --profile platform config` validates
- [x] Lint passes
- [x] Roadmap updated to `done`
- [x] Spec summary appended to docs/spec-summaries.md
