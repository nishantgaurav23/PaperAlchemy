# Spec S1.4 — CI/CD Setup

## Overview
GitHub Actions CI pipeline for PaperAlchemy. Runs on every push and pull request: linting (Ruff), type checking (mypy), unit tests (pytest), and Docker image build verification. This is the foundation CI — production CI/CD with deployment is covered in S11.5.

## Dependencies
- S1.1 (Dependency declaration) — `pyproject.toml` with UV lockfile for dependency installation

## Target Location
- `.github/workflows/ci.yml` — Main CI workflow

## Functional Requirements

### FR-1: CI Workflow Triggers
- **What**: Workflow runs on push to `main` and on all pull requests
- **Triggers**: `push` to `main`, `pull_request` to `main`
- **Edge cases**: Skip CI on docs-only changes (*.md files) using path filters

### FR-2: Lint Job (Ruff)
- **What**: Run Ruff linter and formatter check on Python code
- **Steps**: Install UV, install dependencies, run `ruff check src/ tests/`, run `ruff format --check src/ tests/`
- **Inputs**: Python 3.12
- **Outputs**: Pass/fail status
- **Edge cases**: Must respect `pyproject.toml` Ruff config (line-length 130, excluded dirs)

### FR-3: Type Check Job (mypy)
- **What**: Run mypy type checker on source code
- **Steps**: Install UV, install dependencies, run `mypy src/`
- **Inputs**: Python 3.12
- **Outputs**: Pass/fail status
- **Edge cases**: Must respect `pyproject.toml` mypy config (pydantic plugin)

### FR-4: Test Job (pytest)
- **What**: Run pytest with coverage reporting
- **Steps**: Install UV, install dev dependencies, run `pytest tests/ --cov=src --cov-report=xml`
- **Inputs**: Python 3.12
- **Outputs**: Test results, coverage report (XML)
- **Edge cases**: Tests must work without external services (all mocked); uses `.env.test`

### FR-5: Docker Build Job
- **What**: Verify the Docker image builds successfully (no push)
- **Steps**: Set up Docker Buildx, build image with `--load` flag
- **Inputs**: Dockerfile
- **Outputs**: Pass/fail build status
- **Edge cases**: Uses BuildKit cache for faster builds

### FR-6: UV Dependency Caching
- **What**: Cache UV dependencies across workflow runs for speed
- **Steps**: Cache `~/.cache/uv` directory keyed on `uv.lock` hash
- **Edge cases**: Cache invalidation on lockfile change

## Tangible Outcomes
- [ ] `.github/workflows/ci.yml` exists and is valid YAML
- [ ] CI triggers on push to main and pull requests
- [ ] Lint job runs Ruff check and format check
- [ ] Type check job runs mypy on src/
- [ ] Test job runs pytest with coverage
- [ ] Docker build job verifies image builds
- [ ] UV cache is configured for fast dependency installs
- [ ] Jobs run in parallel where possible (lint, type-check, test in parallel; docker-build after test)
- [ ] Python 3.12 is used across all jobs
- [ ] No hardcoded secrets in workflow file

## Test-Driven Requirements

### Tests to Write First
1. `test_ci_workflow_exists`: `.github/workflows/ci.yml` exists
2. `test_ci_workflow_valid_yaml`: Workflow file parses as valid YAML
3. `test_ci_workflow_triggers`: Has push and pull_request triggers on main
4. `test_ci_workflow_has_lint_job`: Contains a lint job with ruff check + format check
5. `test_ci_workflow_has_type_check_job`: Contains a type-check job with mypy
6. `test_ci_workflow_has_test_job`: Contains a test job with pytest and coverage
7. `test_ci_workflow_has_docker_build_job`: Contains a docker-build job
8. `test_ci_workflow_python_version`: All Python jobs use 3.12
9. `test_ci_workflow_uv_caching`: UV cache step is configured
10. `test_ci_workflow_jobs_parallel`: Lint, type-check, test jobs have no inter-dependencies

### Mocking Strategy
- No mocking needed — these are structural/config tests
- Parse YAML to validate workflow structure
- No GitHub Actions runner required for tests

### Coverage
- All workflow jobs validated for required steps
- Trigger configuration validated
- Caching configuration validated
