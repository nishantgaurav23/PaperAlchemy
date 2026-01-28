"""arXiv client factory."""                                                                                                      
                                                                                                                                   
from functools import lru_cache                                                                                                  
                                                                                                                                
from src.config import get_settings                                                                                              
from src.services.arxiv.client import ArxivClient                                                                                
                                                                                                                                
                                                                                                                                
@lru_cache(maxsize=1)                                                                                                            
def make_arxiv_client() -> ArxivClient:                                                                                          
    """                                                                                                                          
    Create and cache arXiv client instance.                                                                                      
                                                                                                                                
    Returns:                                                                                                                     
        ArxivClient instance (singleton)                                                                                         
    """                                                                                                                          
    settings = get_settings()                                                                                                    
                                                                                                                                
    return ArxivClient(                                                                                                          
        base_url=settings.arxiv.base_url,                                                                                        
        rate_limit_delay=settings.arxiv.rate_limit_delay,                                                                        
        max_results=settings.arxiv.max_results,                                                                                  
        search_category=settings.arxiv.category,                                                                                 
        timeout=settings.arxiv.timeout,                                                                                          
        max_retries=settings.arxiv.max_retries,                                                                                  
        retry_delay=settings.arxiv.retry_delay,                                                                                  
        cache_dir=settings.pdf_parser.cache_dir,                                                                                 
    )                                                                                                                            
                                                                                                                                
                                                                                                                                
def reset_arxiv_client_cache() -> None:                                                                                          
    """Reset the cached client instance."""                                                                                      
    make_arxiv_client.cache_clear()