"""Tests for src/config.py — Pydantic settings configuration."""

from __future__ import annotations


class TestPostgresSettings:
    """Test PostgresSettings sub-settings."""

    def test_postgres_fields(self):
        from src.config import PostgresSettings

        s = PostgresSettings(host="dbhost", port=5432, user="usr", password="pw", db="mydb")
        assert s.host == "dbhost"
        assert s.port == 5432
        assert s.user == "usr"
        assert s.password == "pw"
        assert s.db == "mydb"

    def test_postgres_async_url(self):
        from src.config import PostgresSettings

        s = PostgresSettings(host="h", port=5432, user="u", password="p", db="d")
        assert s.url == "postgresql+asyncpg://u:p@h:5432/d"

    def test_postgres_sync_url(self):
        from src.config import PostgresSettings

        s = PostgresSettings(host="h", port=5432, user="u", password="p", db="d")
        assert s.sync_url == "postgresql://u:p@h:5432/d"
        assert "asyncpg" not in s.sync_url

    def test_postgres_env_prefix_override(self, monkeypatch):
        monkeypatch.setenv("POSTGRES__HOST", "custom-host")
        monkeypatch.setenv("POSTGRES__PORT", "9999")

        from src.config import PostgresSettings

        s = PostgresSettings()
        assert s.host == "custom-host"
        assert s.port == 9999
        assert "custom-host:9999" in s.url


class TestOpenSearchSettings:
    """Test OpenSearchSettings sub-settings."""

    def test_opensearch_fields(self):
        from src.config import OpenSearchSettings

        s = OpenSearchSettings(host="http://os:9200")
        assert s.host == "http://os:9200"
        assert s.index_name == "arxiv-papers"
        assert s.chunk_index_suffix == "chunks"
        assert s.vector_dimension == 1024
        assert s.rrf_pipeline_name == "hybrid-rrf-pipeline"


class TestOllamaSettings:
    """Test OllamaSettings sub-settings."""

    def test_ollama_defaults(self):
        from src.config import OllamaSettings

        s = OllamaSettings()
        assert s.host == "localhost"
        assert s.port == 11434
        assert s.default_model == "llama3.2:1b"
        assert s.default_timeout == 300
        assert s.default_temperature == 0.7
        assert s.default_top_p == 0.9

    def test_ollama_url(self):
        from src.config import OllamaSettings

        s = OllamaSettings()
        assert s.url == "http://localhost:11434"

    def test_ollama_url_custom(self, monkeypatch):
        monkeypatch.setenv("OLLAMA__HOST", "ollama-server")
        monkeypatch.setenv("OLLAMA__PORT", "8080")

        from src.config import OllamaSettings

        s = OllamaSettings()
        assert s.url == "http://ollama-server:8080"


class TestGeminiSettings:
    """Test GeminiSettings sub-settings (NEW)."""

    def test_gemini_settings_fields(self):
        from src.config import GeminiSettings

        s = GeminiSettings(api_key="k", model="gemini-2.0-flash", temperature=0.7, max_output_tokens=4096, timeout=60)
        assert s.api_key == "k"
        assert s.model == "gemini-2.0-flash"
        assert s.temperature == 0.7
        assert s.max_output_tokens == 4096
        assert s.timeout == 60

    def test_gemini_env_override(self, monkeypatch):
        monkeypatch.setenv("GEMINI__API_KEY", "my-gemini-key")
        monkeypatch.setenv("GEMINI__MODEL", "gemini-pro")

        from src.config import GeminiSettings

        s = GeminiSettings()
        assert s.api_key == "my-gemini-key"
        assert s.model == "gemini-pro"


class TestRedisSettings:
    """Test RedisSettings sub-settings."""

    def test_redis_defaults(self):
        from src.config import RedisSettings

        s = RedisSettings()
        assert s.host == "localhost"
        assert s.port == 6379
        assert s.password == ""
        assert s.db == 0
        assert s.ttl_hours == 24
        assert s.decode_responses is True

    def test_redis_url_with_password(self):
        from src.config import RedisSettings

        s = RedisSettings(password="secret123")
        assert s.url == "redis://:secret123@localhost:6379/0"

    def test_redis_url_without_password(self):
        from src.config import RedisSettings

        s = RedisSettings(password="")
        assert s.url == "redis://localhost:6379/0"


class TestJinaSettings:
    """Test JinaSettings sub-settings (NEW)."""

    def test_jina_settings_fields(self):
        from src.config import JinaSettings

        s = JinaSettings(api_key="jk")
        assert s.api_key == "jk"
        assert s.model == "jina-embeddings-v3"
        assert s.dimensions == 1024
        assert s.batch_size == 100
        assert s.timeout == 30


class TestRerankerSettings:
    """Test RerankerSettings sub-settings (NEW)."""

    def test_reranker_settings_defaults(self):
        from src.config import RerankerSettings

        s = RerankerSettings()
        assert s.model == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert s.top_k == 5
        assert s.device == "cpu"


class TestLangfuseSettings:
    """Test LangfuseSettings sub-settings."""

    def test_langfuse_defaults(self):
        from src.config import LangfuseSettings

        s = LangfuseSettings()
        assert s.public_key == ""
        assert s.secret_key == ""
        assert s.host == "http://localhost:3000"
        assert s.enabled is False


class TestArxivSettings:
    """Test ArxivSettings sub-settings."""

    def test_arxiv_defaults(self):
        from src.config import ArxivSettings

        s = ArxivSettings()
        assert s.base_url == "https://export.arxiv.org/api/query"
        assert s.rate_limit_delay == 3.0
        assert s.max_results == 100
        assert s.category == "cs.AI"
        assert s.timeout == 30
        assert s.max_retries == 3


class TestChunkingSettings:
    """Test ChunkingSettings sub-settings."""

    def test_chunking_settings_defaults(self):
        from src.config import ChunkingSettings

        s = ChunkingSettings()
        assert s.chunk_size == 600
        assert s.overlap_size == 100
        assert s.min_chunk_size == 100
        assert s.section_based is True


class TestAppSettings:
    """Test AppSettings sub-settings."""

    def test_app_fields(self):
        from src.config import AppSettings

        s = AppSettings(debug=True, log_level="INFO", title="PaperAlchemy", version="0.1.0")
        assert s.debug is True
        assert s.log_level == "INFO"
        assert s.title == "PaperAlchemy"
        assert s.version == "0.1.0"


class TestRootSettings:
    """Test root Settings composition."""

    def test_default_settings_creation(self):
        from src.config import Settings

        s = Settings()
        assert s is not None

    def test_settings_composition(self):
        """Root Settings has all 11 sub-settings accessible."""
        from src.config import Settings

        s = Settings()
        assert hasattr(s, "postgres")
        assert hasattr(s, "opensearch")
        assert hasattr(s, "ollama")
        assert hasattr(s, "gemini")
        assert hasattr(s, "redis")
        assert hasattr(s, "jina")
        assert hasattr(s, "reranker")
        assert hasattr(s, "langfuse")
        assert hasattr(s, "arxiv")
        assert hasattr(s, "chunking")
        assert hasattr(s, "app")

    def test_settings_sub_settings_types(self):
        from src.config import (
            AppSettings,
            ArxivSettings,
            ChunkingSettings,
            GeminiSettings,
            JinaSettings,
            LangfuseSettings,
            OllamaSettings,
            OpenSearchSettings,
            PostgresSettings,
            RedisSettings,
            RerankerSettings,
            Settings,
        )

        s = Settings()
        assert isinstance(s.postgres, PostgresSettings)
        assert isinstance(s.opensearch, OpenSearchSettings)
        assert isinstance(s.ollama, OllamaSettings)
        assert isinstance(s.gemini, GeminiSettings)
        assert isinstance(s.redis, RedisSettings)
        assert isinstance(s.jina, JinaSettings)
        assert isinstance(s.reranker, RerankerSettings)
        assert isinstance(s.langfuse, LangfuseSettings)
        assert isinstance(s.arxiv, ArxivSettings)
        assert isinstance(s.chunking, ChunkingSettings)
        assert isinstance(s.app, AppSettings)


class TestGetSettings:
    """Test get_settings() factory function."""

    def test_get_settings_returns_settings(self):
        from src.config import Settings, get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_returns_same_instance(self):
        from src.config import get_settings

        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_get_settings_cache_clear(self):
        from src.config import get_settings

        get_settings.cache_clear()
        s1 = get_settings()
        get_settings.cache_clear()
        s2 = get_settings()
        # After cache clear, a new instance is created
        assert s1 is not s2
