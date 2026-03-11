# Checklist — Spec S1.4: CI/CD Setup

## Phase 1: Setup & Dependencies
- [x] Verify S1.1 is "done" (pyproject.toml + uv.lock exist)
- [x] Review existing `.github/workflows/` for patterns

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_ci_cd_setup.py`
- [x] Write test_ci_workflow_exists
- [x] Write test_ci_workflow_valid_yaml
- [x] Write test_ci_workflow_has_name
- [x] Write test_ci_workflow_triggers (push + PR to main)
- [x] Write test_ci_workflow_has_lint_job
- [x] Write test_lint_job_runs_check_and_format
- [x] Write test_ci_workflow_has_type_check_job
- [x] Write test_ci_workflow_has_test_job
- [x] Write test_test_job_has_coverage
- [x] Write test_ci_workflow_has_docker_build_job
- [x] Write test_ci_workflow_python_version
- [x] Write test_ci_workflow_uv_caching
- [x] Write test_ci_workflow_jobs_parallel
- [x] Write test_docker_build_runs_on_ubuntu
- [x] Run tests — expect failures (Red): 15 failed

## Phase 3: Implementation
- [x] Create `.github/workflows/ci.yml`
- [x] Implement lint job (Ruff check + format --check)
- [x] Implement type-check job (mypy src/)
- [x] Implement test job (pytest + coverage XML)
- [x] Implement docker-build job (Buildx, no push)
- [x] Configure UV caching (astral-sh/setup-uv with enable-cache)
- [x] Configure parallel jobs (lint, type-check, test independent; docker-build needs test)
- [x] Run tests — expect pass (Green): 15 passed

## Phase 4: Integration
- [x] Run lint (`ruff check tests/`): All checks passed
- [x] Run full test suite: 62 passed

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S1.4_cicd.ipynb`
- [x] Update `roadmap.md` status to "done"
- [x] Append spec summary to `docs/spec-summaries.md`
