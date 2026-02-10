"""
PaperAlchemy — FastAPI Application Entry Point.

Why it's needed:
    This is the single entry point that boots the entire application.
    It initializes all services (database, OpenSearch), registers all API
    routers, and handles graceful shutdown. Without a lifespan manager,
    services would be created lazily on first request, causing slow first
    responses and potential race conditions.

What it does:
    - lifespan(): Async context manager that runs on startup/shutdown:
      1. Loads settings from environment
      2. Creates database connection pool
      3. Creates OpenSearch client and sets up indices (if not exist)
      4. Stores all services on app.state for dependency injection
      5. On shutdown: tears down database connections
    - Registers routers: ping (/api/v1/health) and search (/api/v1/search)
    - Simple /health endpoint for Docker healthcheck probes
    - CORS middleware allows all origins (development mode)

How it helps:
    - Services are initialized once at startup, not per-request
    - app.state makes services available to all routers via dependency injection
    - Graceful shutdown prevents connection leaks
    - Docker healthcheck uses /health (fast, no dependencies) while
      /api/v1/health checks all services (detailed, for monitoring)
"""                                              
                                                                                                     
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI                                                                        
from fastapi.middleware.cors import CORSMiddleware                                                 
                                                                                                    
# from src.config import settings
from src.config import get_settings
from src.db.factory import make_database
from src.routers import ping, search, hybrid_search
from src.routers.ask import ask_router, stream_router
from src.services.opensearch.factory import make_opensearch_client
from src.services.embeddings.factory import make_embeddings_service
from src.services.ollama.factory import make_ollama_client
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.cache.factory import make_cache_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes and tears down servies on startup/shutdown.
    """
    logger.info("Starting PaperAlchemy API...")

    # Load settings
    settings = get_settings()
    app.state.settings = settings

    # Initialize database
    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Initialize OpenSearch
    opeansearch_client = make_opensearch_client()
    app.state.opensearch_client = opeansearch_client

    if opeansearch_client.health_check():
        logger.info("OpenSearch connected successfully")

        # Setup index (create if not exists)
        setup_results = opeansearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("OpenSearch index created")
        else:
            logger.info("OpenSearch index already exists")

        # Log document count
        try:
            stats = opeansearch_client.get_index_stats()
            doc_count = stats.get("document_count", 0)
            logger.info(f"OpenSearch ready: {doc_count} documents indexed")
        except Exception:
            logger.info("OpenSearch index ready (stats unavailable)")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")

    # Future weeks will add:                                                                                                     
    # app.state.arxiv_client = make_arxiv_client()                                                                               
    # app.state.pdf_parser = make_pdf_parser_service()                                                                           
    app.state.embeddings_service = make_embeddings_service()
    logger.info("Jina embeddings service initialized")

    # Initialize Ollama client
    app.state.ollama_client = make_ollama_client(settings)
    logger.info(f"Ollama client initialized (model: {settings.ollama.default_model}, url: {settings.ollama.url})")

    # Initialize Langfuse tracer
    app.state.langfuse_tracer = make_langfuse_tracer(settings)
    logger.info(f"Langfuse tracer intialized (enabled={settings.langfuse.enabled})")
    
    # Initalize Redis cache client
    app.state.cache_client = await make_cache_client(settings)
    logger.info(f"Cache client initialized (available={app.state.cache_client is not None})")

    logger.info("PaperAlchemy API ready")
    yield

    # Cleanup
    if app.state.langfuse_tracer:
        app.state.langfuse_tracer.shutdown()
    if app.state.cache_client and app.state.cache_client._redis:
        await app.state.cache_client._redis.aclose()
    database.teardown()
    logger.info("PaperAlchemy API shutdown completed.")
                                                                                                    
app = FastAPI(                                                                                     
    title="PaperAlchemy",                                                                          
    description="Transform Academic Papers into Knowledge Gold - Production RAG System",           
    version="0.1.0",
    lifespan=lifespan                                                                               
)                                                                                                  
                                                                                                    
# CORS middleware                                                                                  
app.add_middleware(                                                                                
    CORSMiddleware,                                                                                
    allow_origins=["*"],                                                                           
    allow_credentials=True,                                                                        
    allow_methods=["*"],                                                                           
    allow_headers=["*"],                                                                           
)

# Register routers
app.include_router(hybrid_search.router, prefix="/api/v1")
app.include_router(ping.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(ask_router, prefix="/api/v1")
app.include_router(stream_router, prefix="/api/v1")
                                                                                                    
                                                                                                    
@app.get("/")                                                                                      
async def root():                                                                                  
    """Root endpoint."""                                                                           
    return {                                                                                       
        "name": "PaperAlchemy",                                                                    
        "description": "Transform Academic Papers into Knowledge Gold",                            
        "version": "0.1.0",                                                                        
    }                                                                                              
                                                                                                    
                                                                                                    
@app.get("/health")                                                                                
async def simple_health():
    """Simple health check for Docker/load balancer"""
    return {"status": "ok"}

if __name__=="__main__":
    uvicorn.run(app, port=8000, host='0.0.0.0')