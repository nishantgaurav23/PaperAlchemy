# Checklist — Spec S1.3: Docker Infrastructure

## Phase 1: Setup & Dependencies
- [x] Verify S1.1 is "done" (pyproject.toml + uv.lock exist)
- [x] Review existing compose.yml, Dockerfile, Makefile

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_docker_infrastructure.py`
- [x] Write test_compose_file_valid_yaml
- [x] Write test_compose_all_services_defined
- [x] Write test_compose_health_checks
- [x] Write test_compose_network_assignment
- [x] Write test_compose_volume_definitions
- [x] Write test_compose_profiles (core, full, langfuse, dev-tools)
- [x] Write test_compose_api_env_overrides
- [x] Write test_compose_postgres_config
- [x] Write test_compose_opensearch_config
- [x] Write test_dockerfile_valid
- [x] Write test_dockerfile_multi_stage
- [x] Write test_dockerfile_uv_pattern
- [x] Write test_makefile_targets
- [x] Run tests — expect failures (Red): 7 failed

## Phase 3: Implementation
- [x] Update `compose.yml` — add Docker profiles to all services
- [x] Update `compose.yml` — add healthcheck to pgadmin + langfuse-worker
- [x] Update `compose.yml` — verify all 14 services have health checks
- [x] Update `compose.yml` — verify network assignments
- [x] Update `compose.yml` — verify volume definitions
- [x] Verify `Dockerfile` — multi-stage UV build pattern (already correct)
- [x] Update `Makefile` — add `up`, `up-all`, `up-langfuse`, `down`, `down-clean`, `build` targets
- [x] Run tests — expect pass (Green): 22 passed
- [x] Refactor: fix ruff SIM114 lint warning

## Phase 4: Integration
- [x] Run lint (`ruff check tests/`): All checks passed
- [x] Run full test suite: 47 passed

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded production secrets (dev defaults OK)
- [x] Create notebook: `notebooks/specs/S1.3_docker.ipynb`
- [x] Update `roadmap.md` status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
