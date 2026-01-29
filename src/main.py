"""PaperAlchemy - FastAPI Application Entry Point."""                                              
                                                                                                     
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI                                                                        
from fastapi.middleware.cors import CORSMiddleware                                                 
                                                                                                    
# from src.config import settings
from src.config import get_settings
from src.db.factory import make_database
from src.routers import ping, search
from src.services.opensearch.factory import make_opensearch_client 

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
    # app.state.embeddings_service = make_embeddings_service()                                                                   
    # app.state.ollama_client = make_ollama_client()                                                                             
    # app.state.langfuse_tracer = make_langfuse_tracer()                                                                         
    # app.state.cache_client = make_cache_client(settings) 

    logger.info("PaperAlchemy API ready")
    yield

    # Cleanup
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
app.include_router(ping.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
                                                                                                    
                                                                                                    
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