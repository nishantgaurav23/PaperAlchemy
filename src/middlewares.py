"""
Why it's needed: 
    Every incoming HTTP request should be logged with its method, path, and any errors. This is esseential
    for  debugging production issues - when a uuser reports "search is broken", you can check logs to see exactly failed and 
    why.

How it helps:
    Right now, FastAPI only logs the HTTP status code. These helpers add structured logging that integrates with the
    logging framework we set up in main.py. Thsi will extended into a full BaseHTTPMiddleware with request       
  timing, correlation IDs, and Langfuse trace integration.

PaperAlchemy Request Logging Middleware.

Provides structured logging functions for HTTP request/response tracking.
Currently implemented as standalone helpers called from routers.

Future Enhancement (Week 5+):                                                                                                    
    These will be integrated into a BaseHTTPMiddleware subclass that                                                               
    automatically logs every request with:                                                                                         
    - Request timing (duration in ms)                                                                                              
    - Correlation ID (X-Request-ID header)                                                                                         
    - Langfuse trace integration                                                                                                   
    - Error classification and alerting                                                                                            
                                                                                                                                   
Usage in routers:                                                                                                                
    from src.middlewares import log_request, log_error                                                                           
                                                                                                                                
    @router.get("/search")                                                                                                       
    async def search(q: str):                                                                                                    
        log_request("GET", "/search")                                                                                            
        try:                                                                                                                     
            ...                                                                                                                  
        except Exception as e:                                                                                                   
            log_error(str(e), "GET", "/search")                                                                                  
            raise   
"""
import logging
# Logger inherits that format configured in main.py:
logger = logging.getLogger(__name__)

def log_request(method: str, path: str) -> None:
    """Log an incoming HTTP request.
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE).
        path: Request path (e.g., "/api/v1/search").

    Logs at INFO level. In production, this helps track
    request patterns, identiify hot endpoints, and detect
    unusual traffic spikes.
    """
    logger.info(f"{method} {path}")



def log_error(error: str, method: str, path: str) -> None:
    """Log an error that occured during request handling.
    
    Args:
        error: Error message or exception string.
        method: HTTP method of the failed request.
        path: Request path where the error occured.
    
    Logs at Error level. These entries trigger alerts in
    production monitoring (furure Langfuse integration).    
    """
    logger.error(f"Error in {method} {path}: {error}")
