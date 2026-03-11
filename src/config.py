"""Centralized application configuration using Pydantic Settings.

Each service gets its own BaseSettings subclass with env_prefix (double underscore).
The root Settings class composes all sub-settings into one object.

Usage:
    from src.config import get_settings
    settings = get_settings()
    print(settings.postgres.url)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES__")

    host: str = "localhost"
    port: int = 5433
    user: str = "paperalchemy"
    password: str = "paperalchemy_secret"
    db: str = "paperalchemy"

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def sync_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class OpenSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENSEARCH__")

    host: str = "http://localhost:9201"
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"
    vector_dimension: int = 1024
    vector_space_type: str = "cosinesimil"
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA__")

    host: str = "localhost"
    port: int = 11434
    default_model: str = "llama3.2:1b"
    default_timeout: int = 300
    default_temperature: float = 0.7
    default_top_p: float = 0.9

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEMINI__")

    api_key: str = ""
    model: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_output_tokens: int = 4096
    timeout: int = 60


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS__")

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    ttl_hours: int = 24
    decode_responses: bool = True

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class JinaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JINA__")

    api_key: str = ""
    model: str = "jina-embeddings-v3"
    dimensions: int = 1024
    batch_size: int = 100
    timeout: int = 30


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RERANKER__")

    model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    top_k: int = 5
    device: str = "cpu"
    provider: str = "local"  # "local" or "cohere"
    cohere_api_key: str = ""


class LangfuseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LANGFUSE__")

    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:3000"
    enabled: bool = False


class ArxivSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARXIV__")

    base_url: str = "https://export.arxiv.org/api/query"
    rate_limit_delay: float = 3.0
    max_results: int = 100
    category: str = "cs.AI"
    timeout: int = 30
    max_retries: int = 3


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHUNKING__")

    chunk_size: int = 600
    overlap_size: int = 100
    min_chunk_size: int = 100
    section_based: bool = True


class PDFParserSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PDF_PARSER__")

    max_pages: int = 30
    max_file_size_mb: int = 50
    timeout: int = 120


class HyDESettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HYDE__")

    enabled: bool = True
    max_tokens: int = 300
    temperature: float = 0.3
    timeout: int = 30


class MultiQuerySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MULTI_QUERY__")

    enabled: bool = True
    num_queries: int = 3
    temperature: float = 0.7
    max_tokens: int = 300
    rrf_k: int = 60


class RetrievalPipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_PIPELINE__")

    multi_query_enabled: bool = True
    hyde_enabled: bool = True
    reranker_enabled: bool = True
    parent_expansion_enabled: bool = True
    retrieval_top_k: int = 20
    final_top_k: int = 5


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP__")

    debug: bool = True
    log_level: str = "INFO"
    title: str = "PaperAlchemy"
    version: str = "0.1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres: PostgresSettings = PostgresSettings()
    opensearch: OpenSearchSettings = OpenSearchSettings()
    ollama: OllamaSettings = OllamaSettings()
    gemini: GeminiSettings = GeminiSettings()
    redis: RedisSettings = RedisSettings()
    jina: JinaSettings = JinaSettings()
    reranker: RerankerSettings = RerankerSettings()
    langfuse: LangfuseSettings = LangfuseSettings()
    arxiv: ArxivSettings = ArxivSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    pdf_parser: PDFParserSettings = PDFParserSettings()
    hyde: HyDESettings = HyDESettings()
    multi_query: MultiQuerySettings = MultiQuerySettings()
    retrieval_pipeline: RetrievalPipelineSettings = RetrievalPipelineSettings()
    app: AppSettings = AppSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
