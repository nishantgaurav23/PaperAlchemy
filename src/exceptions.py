"""
Why it's needed:
    Right now , when something failes (OpenSearch down, paper not found, PDF parsing error), the code raises generic `Exception`
    which makes error handling imprecise. Custom exceptions let you catch specific failure types and responds appropiately 
    - return 404 for PaperNotFound, retry on ArxivAPIRateLimitError, alert on OllamaConnectionError, etc.

How it helps:
    Every service layer (database, OpenSearch, PDF parser, LLM) gets its own exception hierarchy. This means routers can do
    except PaperNotFound: return 404 instead of caching everything blindly. It also makes debugging easier - stack traces
    immediately tell you which subsystem failed.
                                                                                                                            
PaperAlchemy Custom Exception Hierarchy.                                                                                         
                                                                                                                                
Organized by subsystem so each service layer has its own exception tree.                                                         
This enables precise error handling in routers and services:                                                                     
- Repository errors  → 404/500 responses                                                                                       
- Parsing errors     → retry or skip logic                                                                                     
- API errors         → rate-limit backoff, timeout retry                                                                       
- LLM errors         → fallback to cached response                                                                             
- Config errors      → fail-fast on startup                                                                                    
                                                                                                                                
Hierarchy:                                                                                                                       
Exception                                                                                                                      
├── RepositoryException        (database layer)                                                                                
│   ├── PaperNotFound          (SELECT returned no rows)                                                                       
│   └── PaperNotSaved          (INSERT/UPDATE failed)                                                                          
├── ParsingException           (document parsing layer)                                                                        
│   └── PDFParsingException    (PDF-specific parsing)                                                                          
│       └── PDFValidationError (file isn't a valid PDF)                                                                        
├── PDFDownloadException       (HTTP download layer)                                                                           
│   └── PDFDownloadTimeoutError                                                                                                
├── PDFCacheException          (local PDF cache)                                                                               
├── OpenSearchException        (search engine layer)                                                                           
├── ArxivAPIException          (arXiv external API)                                                                            
│   ├── ArxivAPITimeoutError   (request timed out)                                                                             
│   ├── ArxivAPIRateLimitError (429 from arXiv)                                                                                
│   └── ArxivParseError        (malformed XML response)                                                                        
├── MetadataFetchingException  (ingestion pipeline)                                                                            
│   └── PipelineException      (DAG/step failure)                                                                              
├── LLMException               (language model layer)                                                                          
│   └── OllamaException        (Ollama-specific)                                                                               
│       ├── OllamaConnectionError (can't reach Ollama)                                                                         
│       └── OllamaTimeoutError    (generation timed out)                                                                       
└── ConfigurationError         (invalid settings at startup)                                                                          
"""

# =============================================================                                                                  
# Database / Repository Exceptions                                                                                               
# ============================================================= 

class RepositoryException(Exception):
    """
    Base Exception for all database repository operations.

    Raised when CRUD operations on the papers table fail.
    Routers should catch this to return 500 Internal Server Error.
    """

class PaperNotFound(RepositoryException):
    """
    Raised when a SELECT query returns no matching paper.

    Typically caught in routers to return HTTP 404.
    Example: GET /papers/{arxiv_id} where arxiv_id doesn't exist.
    """

class PaperNotSaved(RepositoryException):
    """
    Raised when an INSERT or UPDATE fails to persist a paper.

    Could happen due to unique constraint violations (duplicate arxiv_id),
    connection drops, or transcation rollbacks
    """

# =============================================================                                                                  
# PDF Parsing Exceptions                                                                                                         
# ============================================================= 

class ParsingException(Exception):
    """
    Base exception for all document parsing operations.

    Covers PDF, HTML, or any future document format parsing.
    """

class PDFParsingException(ParsingException):
    """
    Raised when Docling fails to extract content from a PDF.

    Common causes: corrupted PDF, unsopported encoding, timeout.
    The ingestion pipeline catches this to mark papers as 'failed'.
    """

class PDFValidationError(PDFParsingException):
    """
    Raised when a file fails PDF format validation before parsing.

    Checks: file size > 0, valid PDF header (%PDF-), not encrypted.
    Failing validation skips the expensive Docling parsing step.
    """

# =============================================================                                                                  
# PDF Download Exceptions                                                                                                        
# =============================================================

class PDFDownloadException(Exception):
    """
    Raised when downloading a PDF for arXiv fails.

    Covers HTTP error (403, 500), network timeouts, and SSL errors.
    """

class PDFDownloadTimeoutError(PDFDownloadException):
    """
    Raised when a PAD download exceeds the confifured timeout.

    Default timeout is set in config. The pipeline can retry with
    exponential backoff when this occurs.
    """ 

# =============================================================                                                                  
# PDF Cache Exceptions                                                                                                           
# =============================================================

class PDFCacheException(Exception):
    """
    Raised for local PDF file cache errors.

    Covers: disk full, permission denied, corrupted cache file.
    Non-fatal - the pipeline falls back to re-downloading
    """

# =============================================================                                                                  
# OpenSearch Exceptions                                                                                                          
# =============================================================   

class OpenSearchException(Exception):
    """
    Raised for OpenSearch client and indexing errors.

    Covers: connection refused, index not found, mapping conflicts,
    bulk indexing failures, and query syntax errors.
    Routers catch this to return 503 servie Unavailable.
    """

# =============================================================                                                                  
# ArXiv API Exceptions                                                                                                           
# =============================================================  

class ArxivAPIException(Exception):
    """
    Base exception for all arXiv API interactions.

    The arXiv API has strict rate limits (1 request per 3 seconds)
    and can return malformed XML. These exceptions help the pipeline
    handle each failure mode differently.
    """

class ArxivAPITimeoutErrorException(ArxivAPIException):
    """Raised when an arXiv API request exceeds the timeout.

    The pipeline retries with exponential backoff (3s, 6s, 12s).
    """

class ArxivAPIRateLimitError(ArxivAPIException):
    """Raised when arXiv returns HTTP 429 (Too Many Requests).

    The pipline must wait before retrying. arXiv recommends
    a minimum 3-second delay between requests.
    """

class ArxivParseError(ArxivAPIException):
    """Raised when the arXiv API response XML cannot be parsed.
    
    Happens when arXiv returns HTML error pages instead of Atom XML,
    or when the feed structure changes unexpetedly.
    """

 # =============================================================                                                                  
# Metadata / Pipeline Exceptions                                                                                                 
# =============================================================

class MetadataFetchingException(Exception):
    """Raised during the metadata fetching pipeline.
    
    Covers the full ingestion flow: arXiv fetch -> parse -> store.
    """

class PipelineException(MetadataFetchingException):
    """Raised when an Airflow DAG step or pipeline stage fails.

    Includes context about which step failed and partial results
    so the pipeline can resume from the last successful step.
    """ 

# =============================================================                                                                  
# LLM / Ollama Exceptions                                                                                                        
# =============================================================

class LLMException(Exception):
    """Base exception for all language model operations.
    
    Covers local (Ollama) and future remote (OpenAI) LLM providers.
    """

class OllamaException(LLMException):
    """Raised for Ollama-specific service errors.
    
    Covers: model not loaded, inference errros, malformed responses.
    """

class OllamaConnectionError(OllamaException):
    """Raised when the Ollama servie is unreachable.
    
    Common causes: Docker container not running, wrong port,
    network partition. The router returns 503 with a helpful message.
    """

class OllamaTimeoutError(OllamaException):
    """Raised when Ollama generation exceeds the timeout.
    
    Large prompts or complex queries can cause this. The router
    can retry with a shorter context or return a partial response.
    """

# =============================================================                                                                  
# Configuration Exceptions                                                                                                       
# ============================================================= 

class ConfigurationError(Exception):
    """Raised when application settings are invalid at startup.
    
    Examples: missing requires env vars, invalid database URL,
    OpenSearch host unreachable. Causes fast failure before
    serving any requests.
    """

