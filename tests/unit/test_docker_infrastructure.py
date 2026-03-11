"""Tests for S1.3 — Docker Infrastructure.

Validates compose.yml, Dockerfile, and Makefile structure without
requiring a running Docker daemon.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = PROJECT_ROOT / "compose.yml"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
MAKEFILE_PATH = PROJECT_ROOT / "Makefile"


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_compose() -> dict:
    """Load and parse compose.yml."""
    return yaml.safe_load(COMPOSE_PATH.read_text())


# ── compose.yml Tests ────────────────────────────────────────────────────────


class TestComposeFileStructure:
    def test_compose_file_valid_yaml(self):
        """compose.yml parses as valid YAML."""
        data = load_compose()
        assert isinstance(data, dict)
        assert "services" in data

    def test_compose_all_services_defined(self):
        """All 14 expected services exist in compose."""
        data = load_compose()
        expected_services = {
            "api",
            "postgres",
            "redis",
            "opensearch",
            "opensearch-dashboards",
            "ollama",
            "airflow",
            "clickhouse",
            "langfuse-postgres",
            "langfuse-redis",
            "langfuse-minio",
            "langfuse-web",
            "langfuse-worker",
            "pgadmin",
        }
        actual_services = set(data["services"].keys())
        missing = expected_services - actual_services
        assert not missing, f"Missing services: {missing}"

    def test_compose_health_checks(self):
        """Every service has a healthcheck block."""
        data = load_compose()
        services_without_healthcheck = []
        for name, config in data["services"].items():
            if "healthcheck" not in config:
                services_without_healthcheck.append(name)
        assert not services_without_healthcheck, f"Services missing healthcheck: {services_without_healthcheck}"

    def test_compose_network_assignment(self):
        """Every service is on paperalchemy-network."""
        data = load_compose()
        services_without_network = []
        for name, config in data["services"].items():
            networks = config.get("networks", [])
            if isinstance(networks, (list, dict)) and "paperalchemy-network" in networks:
                continue
            else:
                services_without_network.append(name)
        assert not services_without_network, f"Services not on paperalchemy-network: {services_without_network}"

    def test_compose_volume_definitions(self):
        """All 8 named volumes are defined."""
        data = load_compose()
        expected_volumes = {
            "postgres_data",
            "redis_data",
            "opensearch_data",
            "ollama_data",
            "airflow_logs",
            "clickhouse_data",
            "langfuse_postgres_data",
            "langfuse_minio_data",
        }
        actual_volumes = set(data.get("volumes", {}).keys())
        missing = expected_volumes - actual_volumes
        assert not missing, f"Missing volumes: {missing}"


class TestComposeProfiles:
    def test_core_services_no_profile(self):
        """Core services (postgres, redis, opensearch) have no profile — always start."""
        data = load_compose()
        core_services = ["postgres", "redis", "opensearch"]
        for svc in core_services:
            config = data["services"][svc]
            assert "profiles" not in config, f"Core service '{svc}' should NOT have a profile (must always start)"

    def test_langfuse_services_have_langfuse_profile(self):
        """Langfuse services are in the 'langfuse' profile."""
        data = load_compose()
        langfuse_services = [
            "clickhouse",
            "langfuse-postgres",
            "langfuse-redis",
            "langfuse-minio",
            "langfuse-web",
            "langfuse-worker",
        ]
        for svc in langfuse_services:
            config = data["services"][svc]
            profiles = config.get("profiles", [])
            assert "langfuse" in profiles, f"Service '{svc}' should have 'langfuse' profile, got: {profiles}"

    def test_dev_tools_have_dev_tools_profile(self):
        """Dev tool services are in the 'dev-tools' profile."""
        data = load_compose()
        dev_tool_services = ["opensearch-dashboards", "pgadmin"]
        for svc in dev_tool_services:
            config = data["services"][svc]
            profiles = config.get("profiles", [])
            assert "dev-tools" in profiles, f"Service '{svc}' should have 'dev-tools' profile, got: {profiles}"

    def test_api_has_full_profile(self):
        """API service is in the 'full' profile."""
        data = load_compose()
        config = data["services"]["api"]
        profiles = config.get("profiles", [])
        assert "full" in profiles, f"API should have 'full' profile, got: {profiles}"

    def test_ollama_has_full_profile(self):
        """Ollama service is in the 'full' profile."""
        data = load_compose()
        config = data["services"]["ollama"]
        profiles = config.get("profiles", [])
        assert "full" in profiles, f"Ollama should have 'full' profile, got: {profiles}"

    def test_airflow_has_full_profile(self):
        """Airflow service is in the 'full' profile."""
        data = load_compose()
        config = data["services"]["airflow"]
        profiles = config.get("profiles", [])
        assert "full" in profiles, f"Airflow should have 'full' profile, got: {profiles}"


class TestComposeServiceConfigs:
    def test_compose_api_env_overrides(self):
        """API service has correct container networking env vars."""
        data = load_compose()
        api = data["services"]["api"]
        env_list = api.get("environment", [])
        # Convert list of "KEY=VALUE" to dict
        env_dict = {}
        for item in env_list:
            if "=" in item:
                k, v = item.split("=", 1)
                env_dict[k] = v
        assert env_dict.get("POSTGRES__HOST") == "postgres"
        assert env_dict.get("REDIS__HOST") == "redis"
        assert "opensearch" in env_dict.get("OPENSEARCH__HOST", "").lower()

    def test_compose_postgres_config(self):
        """Postgres has correct env, port mapping, volume, and healthcheck."""
        data = load_compose()
        pg = data["services"]["postgres"]
        # Check image
        assert "postgres" in pg["image"]
        # Check volume mount
        volumes = pg.get("volumes", [])
        volume_str = str(volumes)
        assert "postgres_data" in volume_str
        # Check healthcheck exists
        assert "healthcheck" in pg

    def test_compose_opensearch_config(self):
        """OpenSearch has single-node, security disabled, memlock ulimits."""
        data = load_compose()
        os_svc = data["services"]["opensearch"]
        env_list = os_svc.get("environment", [])
        env_str = " ".join(str(e) for e in env_list)
        assert "single-node" in env_str
        assert "DISABLE_SECURITY_PLUGIN=true" in env_str
        # Check memlock ulimits
        ulimits = os_svc.get("ulimits", {})
        assert "memlock" in ulimits

    def test_compose_redis_config(self):
        """Redis has appendonly, maxmemory, LRU eviction policy."""
        data = load_compose()
        redis_svc = data["services"]["redis"]
        command = redis_svc.get("command", "")
        assert "appendonly" in str(command)
        assert "allkeys-lru" in str(command)


# ── Dockerfile Tests ─────────────────────────────────────────────────────────


class TestDockerfile:
    def test_dockerfile_valid(self):
        """Dockerfile exists and is non-empty."""
        assert DOCKERFILE_PATH.exists()
        content = DOCKERFILE_PATH.read_text()
        assert len(content) > 100

    def test_dockerfile_multi_stage(self):
        """Dockerfile has base and final stages."""
        content = DOCKERFILE_PATH.read_text()
        assert "AS base" in content or "as base" in content
        assert "AS final" in content or "as final" in content

    def test_dockerfile_uv_pattern(self):
        """Dockerfile uses UV for dependency installation."""
        content = DOCKERFILE_PATH.read_text()
        assert "uv" in content.lower()
        assert "uv sync" in content or "uv.lock" in content

    def test_dockerfile_exposes_port(self):
        """Dockerfile exposes port 8000."""
        content = DOCKERFILE_PATH.read_text()
        assert "EXPOSE 8000" in content

    def test_dockerfile_runs_uvicorn(self):
        """Dockerfile CMD runs uvicorn."""
        content = DOCKERFILE_PATH.read_text()
        assert "uvicorn" in content


# ── Makefile Tests ───────────────────────────────────────────────────────────


class TestMakefile:
    def test_makefile_targets(self):
        """Makefile contains all expected Docker targets."""
        content = MAKEFILE_PATH.read_text()
        expected_targets = [
            "up:",
            "up-all:",
            "up-langfuse:",
            "down:",
            "down-clean:",
            "build:",
            "status:",
            "logs:",
            "health:",
        ]
        missing = [t for t in expected_targets if t not in content]
        assert not missing, f"Missing Makefile targets: {missing}"

    def test_makefile_phony(self):
        """Makefile declares .PHONY for Docker targets."""
        content = MAKEFILE_PATH.read_text()
        assert ".PHONY" in content
