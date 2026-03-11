"""Tests for S1.4 — CI/CD Setup.

Validates .github/workflows/ci.yml structure without
requiring a running GitHub Actions runner.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_workflow() -> dict:
    """Load and parse ci.yml."""
    return yaml.safe_load(CI_WORKFLOW_PATH.read_text())


# ── Workflow Existence & Structure ───────────────────────────────────────────


class TestCIWorkflowStructure:
    def test_ci_workflow_exists(self):
        """CI workflow file exists at .github/workflows/ci.yml."""
        assert CI_WORKFLOW_PATH.exists(), f"Missing: {CI_WORKFLOW_PATH}"

    def test_ci_workflow_valid_yaml(self):
        """CI workflow parses as valid YAML with expected top-level keys."""
        data = load_workflow()
        assert isinstance(data, dict)
        assert "on" in data or True in data  # YAML parses `on:` as True key
        assert "jobs" in data

    def test_ci_workflow_has_name(self):
        """CI workflow has a descriptive name."""
        data = load_workflow()
        assert "name" in data
        assert len(data["name"]) > 0


# ── Trigger Configuration ────────────────────────────────────────────────────


class TestCIWorkflowTriggers:
    def _get_triggers(self) -> dict:
        data = load_workflow()
        # YAML parses `on:` as boolean True key
        return data.get("on") or data.get(True, {})

    def test_triggers_on_push_to_main(self):
        """Workflow triggers on push to main branch."""
        triggers = self._get_triggers()
        assert "push" in triggers
        push_config = triggers["push"]
        branches = push_config.get("branches", [])
        assert "main" in branches

    def test_triggers_on_pull_request(self):
        """Workflow triggers on pull requests to main."""
        triggers = self._get_triggers()
        assert "pull_request" in triggers
        pr_config = triggers["pull_request"]
        branches = pr_config.get("branches", [])
        assert "main" in branches


# ── Job Definitions ──────────────────────────────────────────────────────────


class TestCIWorkflowJobs:
    def _get_jobs(self) -> dict:
        data = load_workflow()
        return data["jobs"]

    def test_ci_workflow_has_lint_job(self):
        """Workflow contains a lint job with ruff."""
        jobs = self._get_jobs()
        assert "lint" in jobs
        lint_steps = yaml.dump(jobs["lint"].get("steps", []))
        assert "ruff" in lint_steps.lower()

    def test_lint_job_runs_check_and_format(self):
        """Lint job runs both ruff check and ruff format --check."""
        jobs = self._get_jobs()
        lint_steps_str = yaml.dump(jobs["lint"].get("steps", []))
        assert "ruff check" in lint_steps_str
        assert "ruff format" in lint_steps_str

    def test_ci_workflow_has_type_check_job(self):
        """Workflow contains a type-check job with mypy."""
        jobs = self._get_jobs()
        assert "type-check" in jobs
        tc_steps = yaml.dump(jobs["type-check"].get("steps", []))
        assert "mypy" in tc_steps.lower()

    def test_ci_workflow_has_test_job(self):
        """Workflow contains a test job with pytest."""
        jobs = self._get_jobs()
        assert "test" in jobs
        test_steps = yaml.dump(jobs["test"].get("steps", []))
        assert "pytest" in test_steps.lower()

    def test_test_job_has_coverage(self):
        """Test job includes coverage reporting."""
        jobs = self._get_jobs()
        test_steps = yaml.dump(jobs["test"].get("steps", []))
        assert "cov" in test_steps.lower()

    def test_ci_workflow_has_docker_build_job(self):
        """Workflow contains a docker-build job."""
        jobs = self._get_jobs()
        assert "docker-build" in jobs
        docker_steps = yaml.dump(jobs["docker-build"].get("steps", []))
        assert "docker" in docker_steps.lower() or "buildx" in docker_steps.lower()


# ── Python Version ───────────────────────────────────────────────────────────


class TestCIWorkflowPythonVersion:
    def test_ci_workflow_python_version(self):
        """All Python jobs use Python 3.12."""
        data = load_workflow()
        jobs = data["jobs"]
        python_jobs = ["lint", "type-check", "test"]
        for job_name in python_jobs:
            job = jobs[job_name]
            steps_str = yaml.dump(job.get("steps", []))
            assert "3.12" in steps_str, f"Job '{job_name}' should use Python 3.12"


# ── Caching ──────────────────────────────────────────────────────────────────


class TestCIWorkflowCaching:
    def test_ci_workflow_uv_caching(self):
        """UV cache is configured in Python jobs."""
        data = load_workflow()
        jobs = data["jobs"]
        # At least one Python job should have UV caching
        python_jobs = ["lint", "type-check", "test"]
        has_caching = False
        for job_name in python_jobs:
            job = jobs[job_name]
            steps_str = yaml.dump(job.get("steps", []))
            if "cache" in steps_str.lower() and "uv" in steps_str.lower():
                has_caching = True
                break
        assert has_caching, "At least one Python job should have UV caching configured"


# ── Job Parallelism ──────────────────────────────────────────────────────────


class TestCIWorkflowParallelism:
    def test_ci_workflow_jobs_parallel(self):
        """Lint, type-check, and test jobs have no inter-dependencies (run in parallel)."""
        data = load_workflow()
        jobs = data["jobs"]
        parallel_jobs = ["lint", "type-check", "test"]
        for job_name in parallel_jobs:
            job = jobs[job_name]
            needs = job.get("needs", [])
            # These jobs should NOT depend on each other
            for other in parallel_jobs:
                if other != job_name:
                    if isinstance(needs, str):
                        assert needs != other, f"Job '{job_name}' should not depend on '{other}'"
                    elif isinstance(needs, list):
                        assert other not in needs, f"Job '{job_name}' should not depend on '{other}'"

    def test_docker_build_runs_on_ubuntu(self):
        """Docker build job runs on ubuntu-latest."""
        data = load_workflow()
        jobs = data["jobs"]
        docker_job = jobs["docker-build"]
        assert "ubuntu" in docker_job.get("runs-on", "")
