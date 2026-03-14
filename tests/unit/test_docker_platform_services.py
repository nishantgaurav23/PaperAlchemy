"""Tests for S9b.7 — Docker Platform Services.

Validates compose.yml has MinIO, sandbox services, API env vars,
.env.example has new vars, and Makefile has platform target.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = ROOT / "compose.yml"
ENV_EXAMPLE_PATH = ROOT / ".env.example"
MAKEFILE_PATH = ROOT / "Makefile"


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text())


@pytest.fixture(scope="module")
def env_example() -> str:
    return ENV_EXAMPLE_PATH.read_text()


@pytest.fixture(scope="module")
def makefile() -> str:
    return MAKEFILE_PATH.read_text()


# ── R1: MinIO + R2: Sandbox exist ──────────────────────────────────────────


class TestComposeServicesExist:
    def test_minio_service_exists(self, compose: dict) -> None:
        assert "minio" in compose["services"], "MinIO service missing from compose.yml"

    def test_sandbox_service_exists(self, compose: dict) -> None:
        assert "sandbox" in compose["services"], "Sandbox service missing from compose.yml"


# ── R1: MinIO service config ───────────────────────────────────────────────


class TestMinioServiceConfig:
    def test_image(self, compose: dict) -> None:
        svc = compose["services"]["minio"]
        assert "minio/minio" in svc["image"]

    def test_profile(self, compose: dict) -> None:
        svc = compose["services"]["minio"]
        assert "platform" in svc.get("profiles", [])

    def test_ports(self, compose: dict) -> None:
        svc = compose["services"]["minio"]
        ports = svc.get("ports", [])
        port_strs = [str(p) for p in ports]
        assert any("9100" in p for p in port_strs), "MinIO API port 9100 not mapped"
        assert any("9101" in p for p in port_strs), "MinIO Console port 9101 not mapped"

    def test_healthcheck(self, compose: dict) -> None:
        svc = compose["services"]["minio"]
        assert "healthcheck" in svc

    def test_volume(self, compose: dict) -> None:
        svc = compose["services"]["minio"]
        volumes = svc.get("volumes", [])
        assert any("minio_data" in str(v) for v in volumes)

    def test_network(self, compose: dict) -> None:
        svc = compose["services"]["minio"]
        networks = svc.get("networks", [])
        assert "paperalchemy-network" in networks

    def test_bucket_auto_created(self, compose: dict) -> None:
        """MinIO entrypoint should create the paperalchemy bucket."""
        svc = compose["services"]["minio"]
        cmd = svc.get("command", "") or svc.get("entrypoint", "")
        combined = str(cmd) + str(svc.get("command", ""))
        assert "paperalchemy" in combined, "Bucket auto-creation not in entrypoint/command"


# ── R2: Sandbox service config ─────────────────────────────────────────────


class TestSandboxServiceConfig:
    def test_image(self, compose: dict) -> None:
        svc = compose["services"]["sandbox"]
        assert "dind" in svc["image"] or "docker" in svc["image"]

    def test_profile(self, compose: dict) -> None:
        svc = compose["services"]["sandbox"]
        assert "platform" in svc.get("profiles", [])

    def test_privileged(self, compose: dict) -> None:
        svc = compose["services"]["sandbox"]
        assert svc.get("privileged") is True

    def test_healthcheck(self, compose: dict) -> None:
        svc = compose["services"]["sandbox"]
        assert "healthcheck" in svc

    def test_volume(self, compose: dict) -> None:
        svc = compose["services"]["sandbox"]
        volumes = svc.get("volumes", [])
        assert any("sandbox_data" in str(v) for v in volumes)

    def test_network(self, compose: dict) -> None:
        svc = compose["services"]["sandbox"]
        networks = svc.get("networks", [])
        assert "paperalchemy-network" in networks


# ── R3: API service env vars ───────────────────────────────────────────────


class TestApiEnvVars:
    def test_minio_endpoint(self, compose: dict) -> None:
        env = compose["services"]["api"].get("environment", [])
        env_str = str(env)
        assert "MINIO__ENDPOINT" in env_str

    def test_sandbox_host(self, compose: dict) -> None:
        env = compose["services"]["api"].get("environment", [])
        env_str = str(env)
        assert "SANDBOX__HOST" in env_str


# ── R4: .env.example vars ─────────────────────────────────────────────────


class TestEnvExample:
    def test_minio_endpoint(self, env_example: str) -> None:
        assert "MINIO__ENDPOINT" in env_example

    def test_minio_access_key(self, env_example: str) -> None:
        assert "MINIO__ACCESS_KEY" in env_example

    def test_minio_secret_key(self, env_example: str) -> None:
        assert "MINIO__SECRET_KEY" in env_example

    def test_minio_bucket(self, env_example: str) -> None:
        assert "MINIO__BUCKET" in env_example

    def test_sandbox_host(self, env_example: str) -> None:
        assert "SANDBOX__HOST" in env_example

    def test_sandbox_port(self, env_example: str) -> None:
        assert "SANDBOX__PORT" in env_example


# ── R5: Makefile platform target ──────────────────────────────────────────


class TestMakefilePlatformTarget:
    def test_platform_target_exists(self, makefile: str) -> None:
        assert re.search(r"^platform:", makefile, re.MULTILINE), "Makefile missing 'platform' target"

    def test_platform_uses_profile(self, makefile: str) -> None:
        assert "--profile platform" in makefile


# ── Volumes declared ──────────────────────────────────────────────────────


class TestVolumesDeclared:
    def test_minio_data_volume(self, compose: dict) -> None:
        assert "minio_data" in compose.get("volumes", {})

    def test_sandbox_data_volume(self, compose: dict) -> None:
        assert "sandbox_data" in compose.get("volumes", {})
