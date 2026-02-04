"""
Pydantic schemas for Ollama LLM responses.

Why it's needed:
    When using Ollama's structured output feature, the lLM returns JSON
    that must conform to a specific schema. Pydantic models define this 
    schema and validate responses automatically.

What it does:
    - RagResponse: The expected structure for RAG answers
    - Define fields: answer, sources, confidence, citations
    - Provide JSON schema for Ollama's `format` parameter

How it helps:
    - Ollama can constrain output to match this exact structure.
    - Invalid responses are caught by Pydantic validation
    - Fallback parsing extracts what it can from malformed responses

What Each Field Does                                                                                                             
  ┌────────────┬──────────────────────────────────┬──────────────────────────────────────────────────┐                             
  │   Field    │               Type               │                     Purpose                      │                             
  ├────────────┼──────────────────────────────────┼──────────────────────────────────────────────────┤                             
  │ answer     │ str                              │ The actual response text shown to the user       │                             
  ├────────────┼──────────────────────────────────┼──────────────────────────────────────────────────┤                             
  │ sources    │ List[str]                        │ PDF URLs for "View Source" links in UI           │                             
  ├────────────┼──────────────────────────────────┼──────────────────────────────────────────────────┤                             
  │ confidence │ Literal["high", "medium", "low"] │ LLM's self-assessment (useful for UI indicators) │                             
  ├────────────┼──────────────────────────────────┼──────────────────────────────────────────────────┤                             
  │ citations  │ List[str]                        │ Raw arXiv IDs for tracing which papers were used │                             
  └────────────┴──────────────────────────────────┴──────────────────────────────────────────────────┘                             
  ---                                                                                                                              
  Why Structured Output?                                                                                                           
                                                                                                                                   
  Without structured output, the LLM might return:                                                                                 
  Here's what I found about transformers...                                                                                        
                                                                                                                                   
  With structured output (using RAGResponse.model_json_schema()), Ollama constrains the response to:                               
  {                                                                                                                                
    "answer": "Here's what I found about transformers...",                                                                         
    "sources": ["https://arxiv.org/pdf/1706.03762.pdf"],                                                                           
    "confidence": "high",                                                                                                          
    "citations": ["1706.03762"]                                                                                                    
  }                                                                                                                                
                                                                                                                                   
  This makes parsing reliable and enables rich UI features (source links, confidence badges).   
"""

from typing import List, Literal

from pydantic import BaseModel, Field

class RAGResponse(BaseModel):
    """Structured response from Ollama for RAG queries.
    
    Used in two ways:
    1. As Ollama's `format` parameter to constrain JSON output
    2. To Validate and parse the LLM's response

    Fields:
        answer: The generated response text addressing the user's question
        sources: List of arXiv PDF URLs cited in the answer
        confidence: Self-assessed cnfidence level (high/medium/low)
        citations: List of arXiv IDs references (e.g., ["2301.00001", "2302.00002"])
    """

    answer: str = Field(
        ...,
        description="The generated answer based on retrieved paper chunks"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of source PDF URLs (https://arxiv.org/pdf/...)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Model's self-assessed confidence in the answer"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of arXiv IDs cited in the answer"
    )
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Transformers are a neural network architecture introduced in "
                          "'Attention Is All You Need' [arXiv:1706.03762]. They use "
                          "self-attention mechanisms to process sequences in parallel.",
                "sources": [
                    "https://arxiv.org/pdf/1706.03762.pdf",
                    "https://arxiv.org/pdf/1810.04805.pdf"
                ],
                "confidence": "high",
                "citations": ["1706.03762", "1810.04805"]
            }
        }

    
