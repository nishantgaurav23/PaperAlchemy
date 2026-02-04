"""
Pydantic schemas for the RAG /ask and /stream API endpoints.

Why it's needed:
    The /ask endpoint receives user questions and returns LLM-generated answers.
    These schemas define the contract between frontend and backend:
    - What parameters the client can send (query, model, temperature, etc)
    - What response structure the client can expect (answer, sources, etc.)

What it does:
    - AskRequest: Validates incoming questions with optional overides
    - AskResponse: Structures the RAG answer with metadata
    - All generation parameters are optional (fall back to server defaults)

How it helps:
    - FastAPI auto-generates OpenAPI docs from these schemas
    - Invalid requests faile fast with clear validation errors
    - Client now exactly what to send and what to expect back

Key Design Decisions:                                                                                                             
  ┌─────────────────────────────────────┬─────────────────────────────────────────────────────────────┐                            
  │              Decision               │                          Rationale                          │                            
  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────┤                            
  │ model: Optional[str] = None         │ Falls back to settings.ollama.default_model — no hardcoding │                            
  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────┤                            
  │ temperature: Optional[float] = None │ Falls back to settings.ollama.default_temperature           │                            
  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────┤                            
  │ top_k: int = 5                      │ Reasonable default; more chunks = more context but slower   │                            
  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────┤                            
  │ max_length=2000 for query           │ Prevents abuse; long questions are usually poorly formed    │                            
  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────┤                            
  │ ge=1, le=20 for top_k               │ Guard rails: at least 1 chunk, max 20 to limit context size │                            
  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────┤                            
  │ sources: List[str] in response      │ Enables "View Source" functionality in UI                   │                            
  ├─────────────────────────────────────┼─────────────────────────────────────────────────────────────┤                            
  │ model in response                   │ Transparency: user knows which model answered               │                            
  └─────────────────────────────────────┴─────────────────────────────────────────────────────────────┘  
    
How Optional Parameters Work in the Router:
    # In ask.py router:                                                                                                              
     model = request.model or settings.ollama.default_model                                                                           
     temperature = request.temperature or settings.ollama.default_temperature                                                         
     top_p = request.top_p or settings.ollama.default_top_p                                                                           
                                                                                                                                   
     This creates a 3-tier configuration hierarchy:                                                                                   
     1. Request parameter (highest priority) — User explicitly sets it                                                                
     2. Environment variable — Server admin configures default                                                                        
     3. Code default — Fallback if nothing else is set 
"""

from typing import List, Optional

from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    """Request model for RAG question answering.
    
    Required:
        query: The user's question (1-2000 characters)

    Optional (all fack back to server defaults from OllamaSettings):
        top_k: How many chunks to retrieve for context
        use_hybrid: Whether to use hybrid search (BM25 + vector)
        model: Which Ollama model to use for generation
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
        top_p: Nucleus sampling threshold
        categories: Filter search to specific arXiv categories

    The optional parameters allow per-request customization while
    sensible defaults come from environment configuration.
    """

    query: str = Field(
        ...,
        description="The user's questions to answer",
        min_length=1,
        max_length=2000
    )
    top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve for context",
        ge=1,
        le=20
    )
    use_hybrid: bool = Field(
        default=True,
        description="Use hybrid search (BM25 + vector). Fall back to BM25 if embeddings fail."
    )
    model: Optional[str] = Field(
        default=None,
        description="Ollama model to use (e.g., 'llama3.2', 'mistral:7b'). Uses server default if not specified."
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Generation temperature (0.0-1.0). Higher = more creative.",
        ge=0.0,
        le=1.0
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling threshold (0.0-1.0).",
        ge=0.0,
        le=1.0
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Filter by arXiv categories (e.g., ['cs.AI', 'cs.LG])"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the transformers in machine learning?",
                "top_k": 5,
                "use_hybrid": True,
                "model": "llama3.2",
                "temperature": 0.7,
                "categories": ["cs.AI", "cs.LG"]
            }
        }

class AskResponse(BaseModel):
    """Response model for RAG question answering.
    
    Fields:
        query: Echo of the original question (useful for async/logging)
        answer: The LLM-generated response based on retrieved papers
        sources: PDF URLs of papers used as context (for "View Source" links)
        chunks_used: How many chunks were in the context (transparency metric)
        search_mode: Which search strategy was used ('bm25' or 'hybrid')
        model: Which model generated the answer

    The source list let users veriify claims by reading original papers.
    chunks_used helps tune top_k parameter for future queries.
    """

    query: str = Field(
        ...,
        description="The original user question"
    )
    answer: str = Field(
        ...,
        description="Generated answer from the LLM"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="PDF URLs of source papers"
    )
    chunks_used: int = Field(
        ...,
        description="Number of chunks used as context"
    )
    search_mode: str = Field(
        ...,
        description="Search mode used: 'bm25' or 'hybrid'"
    )
    model: str = Field(
        ...,
        description="Model that generated the answer"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are transformers in machine learning?",
                "answer": "Transformers are a neural network architecture introduced in the paper 'Attention Is All You Need'"    
                          "[arXiv:1706.03762]. They revolutionized NLP by using self-attention mechanisms...",                                              
                "sources": [                                                                                                     
                    "https://arxiv.org/pdf/1706.03762.pdf",                                                                      
                    "https://arxiv.org/pdf/1810.04805.pdf"                                                                       
                ],                                                                                                               
                "chunks_used": 5,                                                                                                
                "search_mode": "hybrid",                                                                                         
                "model": "llama3.2"                              
            }
        }