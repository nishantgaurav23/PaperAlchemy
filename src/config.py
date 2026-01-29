"""Application configuration using Pydantic Settings."""                                           
                                                                                                     
from pydantic_settings import BaseSettings, SettingsConfigDict                                     
                                                                                                    
                                                                                                    
class PostgresSettings(BaseSettings):                                                              
    """PostgreSQL database settings."""                                                            
                                                                                                    
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
    """OpenSearch settings."""                                                                     
                                                                                                    
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
    """Ollama LLM settings."""                                                                     
                                                                                                    
    model_config = SettingsConfigDict(env_prefix="OLLAMA__")                                       
                                                                                                    
    host: str = "localhost"                                                                        
    port: int = 11434                                                                              
    model: str = "llama3.2"                                                                        
                                                                                                    
    @property                                                                                      
    def url(self) -> str:                                                                          
        return f"http://{self.host}:{self.port}"                                                   
                                                                                                    
                                                                                                    
class RedisSettings(BaseSettings):                                                                 
    """Redis cache settings."""                                                                    
                                                                                                    
    model_config = SettingsConfigDict(env_prefix="REDIS__")                                        
                                                                                                    
    host: str = "localhost"                                                                        
    port: int = 6379                                                                               
                                                                                                    
                                                                                                    
class LangfuseSettings(BaseSettings):                                                              
    """Langfuse monitoring settings."""                                                            
                                                                                                    
    model_config = SettingsConfigDict(env_prefix="LANGFUSE__")                                     
                                                                                                    
    public_key: str = ""                                                                           
    secret_key: str = ""                                                                           
    host: str = "http://localhost:3000"                                                            
                                                                                                    
                                                                                                    
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
                                                                                                    
class AppSettings(BaseSettings):                                                                   
    """Application settings."""                                                                    
                                                                                                    
    model_config = SettingsConfigDict(env_prefix="APP__")                                          
                                                                                                    
    debug: bool = True                                                                             
    log_level: str = "INFO"                                                                        
                                                                                                    
                                                                                                    
class Settings(BaseSettings):                                                                      
    """Main settings class combining all settings."""                                              
                                                                                                    
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

    # API Keys                                                                                     
    jina_api_key: str = ""                                                                         
                                                                                                    
                                                                                                    
# Global settings instance                                                                         
settings = Settings()

def get_settings() -> Settings:                                                                 
    """Get the global settings instance."""                                                       
    return settings