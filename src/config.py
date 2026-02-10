"""
Centralized application configuration using Pydantic Settings.

Why it's needed:
    Every service (PostgreSQL, OpenSearch, Ollama, Redis, etc.) needs connection
    details. Hardcoding these values would make it impossible to run the same
    code in Docker (where hosts are service names like "postgres") vs locally
    (where hosts are "localhost"). Pydantic Settings reads from environment
    variables and .env files, providing one source of truth with sensible defaults.

What it does:
    - Each service gets its own Settings class with env_prefix (e.g., POSTGRES__)
    - The main Settings class composes all sub-settings into one object
    - Environment variables override defaults: POSTGRES__HOST=postgres overrides
      the default "localhost"
    - Double underscore (__) in env_prefix maps to nested settings in Docker
      compose environment blocks

How it helps:
    - Local development: defaults work out of the box (localhost, default ports)
    - Docker containers: compose.yml sets POSTGRES__HOST=postgres, OPENSEARCH__HOST=http://opensearch:9200
    - Production: .env file or environment variables configure everything
    - Type safety: Pydantic validates types (port must be int, url must be str)

Architecture:
    Settings is a singleton accessed via get_settings(). It's stored on
    app.state during FastAPI lifespan startup and injected into routers
    via the SettingsDep dependency (see dependency.py).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class PostgresSettings(BaseSettings):
    """PostgreSQL database connection settings.

    Used by: src/db/database.py to create SQLAlchemy engine.
    The url property builds the full connection string from parts.
    """                                                            
                                                                                                    
    model_config = SettingsConfigDict(env_prefix="POSTGRES__")                                     
                                                                                                    
    host: str = "localhost"
    port: int = 5433
    user: str = "paperalchemy"                                                                     
    password: str = "paperalchemy_secret"                                                          
    db: str = "paperalchemy"                                                                       
                                                                                                    
    @property                                                                                      
    def url(self) -> str:                                                                          
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"       
                                                                                                    
                                                                                                    
class OpenSearchSettings(BaseSettings):
    """OpenSearch search engine settings.

    Used by: src/services/opensearch/client.py to connect to the cluster.
    The chunk index name is built as: {index_name}-{chunk_index_suffix}
    e.g., "arxiv-papers-chunks" — this is the primary index for search.
    """                                                                     
                                                                                                    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="OPENSEARCH__",
        extra="ignore",
    )                                   
                                                                                                    
    host: str = "http://localhost:9201"  # PaperAlchemy uses port 9201
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"  # Creates: {index_name}-{suffix}
    max_text_size: int = 1000000

    # Vector search settings (for future weeks)
    vector_dimension: int = 1024  # Jina embeddings dimension
    vector_space_type: str = "cosinesimil"

    # Hybrid search settings (for future weeks)
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = 2                                                  
                                                                                                    
                                                                                                    
class OllamaSettings(BaseSettings):                                                                
    """Ollama LLM settings for local language model inference.
    
    Why it's needed:
        Ollama suns locally and provides an API similar to OPenAI's. We need 
        connection details (host/port), default model selection, and generation
        parameters. All values are defaults that can be overridden per-request.

    What it does:
        - host/port: Where Ollama server is running (localhost:11424 by default)
        - default_model: Fallback model when not specified in request
        - default_timeout: How long to wait for LLM responses
        - default_temperature: Controls randomness in generation
        - default_top_p: Nucleus sampling threshold

    How it helps:
        - All configurable via OLLAMA_* environment variables
        - Docker uses OLLAMA_HOST=ollama, local uses localhost
        - Change defaults without code changes via .env file
        - Per request override via AskRequest fields (model, temperature, etc.)

    Configuration hierarchy (highest priority wins):
        1. Per-request parameter (in API call body)
        2. Environment variable (OLLAMA_DEFAULT_MODEL=...)
        3. Code default (defined here)
    """                                                                     
                                                                                                    
    model_config = SettingsConfigDict(env_prefix="OLLAMA__")   

    # Connection settingss                                                                                                                                   
    host: str = "localhost"                                                                        
    port: int = 11434    

    # Generation defaults (all overridable per-request)                                                                          
    default_model: str = "llama3.2:1b" # Ollama model name (e.g., "llama3.2:1b")
    default_timeout: int = 300 # Seconds; LLM generation can b slow
    default_temperature: float = 0.7 # 0.0=deterministic, .0=very random
    default_top_p: float = 0.9 # Nucleus sampling: top 90% probability mass                                                                       
                                                                                                    
    @property                                                                                      
    def url(self) -> str:
        """Build full Ollama API base URL."""                                                                          
        return f"http://{self.host}:{self.port}"                                                   
                                                                                                    
                                                                                                    
class RedisSettings(BaseSettings):
    """Redis cache settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS__")

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0
    ttl_hours: int = 24
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    decode_responses: bool = True                                                                               
                                                                                                    
                                                                                                    
class LangfuseSettings(BaseSettings):
    """Langfuse monitoring settings."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE__")

    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:3000"
    enabled: bool = False
    flush_at: int = 15
    flush_interval: float = 10.0
    debug: bool = False
    max_retries: int = 3
    timeout: int = 10                                                            
                                                                                                    
                                                                                                    
class TelegramSettings(BaseSettings):                                                              
    """Telegram bot settings."""                                                                   
                                                                                                    
    model_config = SettingsConfigDict(env_prefix="TELEGRAM__")                                     
                                                                                                    
    bot_token: str = ""                                                                            

class ArxivSettings(BaseSettings):
    """arXiv API settings."""

    model_config = SettingsConfigDict(env_prefix="ARXIV__")

    base_url: str = "http://export.arxiv.org/api/query"
    rate_limit_delay: float = 3.0 # arXiv requires 3s between requests
    max_results: int = 100
    category: str = "cs.AI"  # Default category: Computer Science - Artificial Intelligence
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0 # Initial retry delay  (exponential backoff)

class PDFParserSettings(BaseSettings):
    """PDF Parser settings."""                                                                     
                                                                                                    
    model_config = SettingsConfigDict(env_prefix="PDF_PARSER__")                                     
                                                                                                    
    max_pages: int = 50
    max_file_size_mb: int = 50                                                                            
    timeout: int = 30
    cache_dir: str = "data/arxiv_pdfs"

class ChunkingSettings(BaseSettings):
    """Text chunking settings for splitting papers into searchable segments.

    Why it's needed:
        Full paper text is too long for embedding models (Jina v3 has a token
        limit) and too coarse for precise retrieval. Chunking splits papers
        into overlapping segments so each chunk can be independently embedded
        and retrieved.

    What it does:
        - chunk_size: Target number of words per chunk (600 balances context
          richness with embedding model limits)
        - overlap_size: Words shared between adjacent chunks (100 words ensures
          sentences split at chunk boundaries are stll retrievable)
        - min_chunk_size: Chunks smaller than this are merged with neighbors
          to avoid low-quality fragments (e.g., a 20 word "Acknowledgments)
        - section_based: When True, respects paper section boundaries
          (Introduction, Methods, etc.) so chunks don't mix unrelated content
    
    How it helps:
        - 600-word chunks ≈ 800 tokens - fits comfortably with Jina's context
        - 100-word overlap means a sentence on a boundary appears in both chunks,
          so it's a;ways retrieval regardless of which chunk matches.
        - Section-aware chunking keeps "Methods" text separate from "Results",
          improving retrieval precision for section-specific queries.
    """

    model_config = SettingsConfigDict(ebv_prefic="CHUNKING__")

    chunk_size: int = 600           # Target words per chunk
    overlap_size: int = 100         # Words to overlap between chunks
    min_chunk_size: int = 100       # Minimum words for a valid chunk
    section_based: bool = True      # Respect paper section boundaries

    
                                                                                                    
class AppSettings(BaseSettings):                                                                   
    """Application settings."""                                                                    
                                                                                                    
    model_config = SettingsConfigDict(env_prefix="APP__")                                          
                                                                                                    
    debug: bool = True                                                                             
    log_level: str = "INFO"                                                                        
                                                                                                    
                                                                                                    
class Settings(BaseSettings):
    """Root settings class composing all sub-settings into one object.

    Access pattern: settings.postgres.url, settings.opensearch.host, etc.
    Loaded once at startup and stored on FastAPI app.state for dependency injection.
    """                                              
                                                                                                    
    model_config = SettingsConfigDict(                                                             
        env_file=".env",                                                                           
        env_file_encoding="utf-8",                                                                 
        extra="ignore"                                                                             
    )                                                                                              
                                                                                                    
    postgres: PostgresSettings = PostgresSettings()                                                
    opensearch: OpenSearchSettings = OpenSearchSettings()                                          
    ollama: OllamaSettings = OllamaSettings()                                                      
    redis: RedisSettings = RedisSettings()                                                         
    langfuse: LangfuseSettings = LangfuseSettings()                                                
    telegram: TelegramSettings = TelegramSettings()                                                
    app: AppSettings = AppSettings() 
    arxiv: ArxivSettings = ArxivSettings()
    pdf_parser: PDFParserSettings = PDFParserSettings()
    chunking: ChunkingSettings = ChunkingSettings()                                                             

    # API Keys                                                                                     
    jina_api_key: str = ""                                                                         
                                                                                                    
                                                                                                    
# Global settings instance                                                                         
settings = Settings()

def get_settings() -> Settings:                                                                 
    """Get the global settings instance."""                                                       
    return settings