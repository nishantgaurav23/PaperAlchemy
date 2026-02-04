"""
RAG prompt construction and response parsing for Ollama.

Why it's needed:
    The quality of RAG answers depends heavily on how we structure the prompt.
    A well-designed prompt tells the LLM exactly how to use the retrieved
    context, cite sources, and format its response.

What it does:
    - RAGPromptBuilder: Constructs promopts with systems instructions + context + query
    - ResponseParser: Extracts structured data from LLM responses
    - Loads system prompt from file (easy to edit without code changes)

How it helps:
    - Separates prompt engineering from application logic
    - System prompt in a text file allows non-developers to tune it
    - Structured output parsing enables rich UI features (sources, confidence)
    - Fallback parsing handles cases where LLM ignores format instructions

Key Components Explained:                                                                                                          
                                                                                                                                   
  1. RAGPromptBuilder                                                                                                              
  ┌────────────────────────────┬──────────────────────────────────────────────────────────────────┐                                
  │           Method           │                             Purpose                              │                                
  ├────────────────────────────┼──────────────────────────────────────────────────────────────────┤                                
  │ _load_system_prompt()      │ Loads from file or uses fallback — editable without code changes │                                
  ├────────────────────────────┼──────────────────────────────────────────────────────────────────┤                                
  │ create_rag_prompt()        │ Builds prompt for natural language responses                     │                                
  ├────────────────────────────┼──────────────────────────────────────────────────────────────────┤                                
  │ create_structured_prompt() │ Adds JSON schema for constrained output                          │                                
  └────────────────────────────┴──────────────────────────────────────────────────────────────────┘                                
  2. ResponseParser                                                                                                                
  ┌───────────────────────────────┬───────────────────────────────────────┐                                                        
  │            Method             │                Purpose                │                                                        
  ├───────────────────────────────┼───────────────────────────────────────┤                                                        
  │ parse_structured_response()   │ Main entry: tries JSON, then fallback │                                                        
  ├───────────────────────────────┼───────────────────────────────────────┤                                                        
  │ _extract_json_fallback()      │ Finds JSON in mixed text              │                                                        
  ├───────────────────────────────┼───────────────────────────────────────┤                                                        
  │ extract_citations_from_text() │ Regex extraction of arXiv IDs         │                                                        
  └───────────────────────────────┴───────────────────────────────────────┘                                                        
  3. Fallback Strategy                                                                                                             
                                                                                                                                   
  LLM Response                                                                                                                     
      │                                                                                                                            
      ▼                                                                                                                            
  Try JSON.parse()  ──success──►  Validate with RAGResponse                                                                        
      │                                    │                                                                                       
      │fail                                │                                                                                       
      ▼                                    ▼                                                                                       
  Extract {.*} regex ──success──►  Validate with RAGResponse                                                                       
      │                                    │                                                                                       
      │fail                                │                                                                                       
      ▼                                    ▼                                                                                       
  Return as plain text answer      Return validated response                                                                       
  with confidence="low"               
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from src.schemas.ollama import RAGResponse

logger = logging.getLogger(__name__)

class RAGPromptBuilder:
    """Builds prompts for RAG question answering.
    The pormpt has three parts:
    1. System instructions (from rag_system.txt) - tells LLM how to behave
    2. Context from papers - the retrieved chunks with arXiv IDs
    3. User's question - what they want answered

    Example output:
        You are an AI assistant specialized in answering questions...

        ### Context from Papers:

        [1. arXiv:230.00001]
        Transformers use self-attention mechanisms to process sequences...

        [2. arXiv:2302.0002]
        The attention mechanism compytes weighted sum of values...

        ### Question:
        What are transformers?

        ### Answer:
        Provide a natural response and cite sources using [arXiv:id] format.
    """

    def __init__(self):
        """Initialize the prompt builder and load system prompt."""
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the text file.
        
        Falls back to a default prompt if file doesn't exists.
        This allows customization without code changes.

        Returns:
            System prompt string
        """
        prompt_file = self.prompts_dir / "rag_system.txt"

        if prompt_file.exists():
            logger.debug(f"Loading system prompt from {prompt_file}")
            return prompt_file.read_text().strip()
        
        # Fallback default if file doesn't exisit
        logger.warning(f"System prompt file not found at {prompt_file}, using default")
        return(
            "You are an AI assistant specialized in answering questions about "
            "academic papers from arXiv. Base your answer STRICTLY on the provided "
            "paper excerpts. If the context doesn't contain enough information to "
            "answer the question, say so honestly."
        )
    
    def create_rag_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create a RAG prompt with query and retrieved chunks.
        
        This is the main prompt format for natural language responses.
        The LLM received context chunks and is asked to synthesize an answer.

        Args:
            query: User's question
            chunks: List of retrieved chunks with metadata from OpenSearch
                    Each chunk should have: chunk_text (or content), arxid_id

        Returns:
            Formatted prompt string ready for Ollama
        """
        prompt = f"{self.system_prompt}\n\n"
        prompt += "### Context from Papers:\n\n"

        for i, chunk in enumerate(chunks, 1):
            # Get the chunk text (handle different field names)
            chunk_text = chunk.get("chunk_text", chunk.get("content", chunk.get("abstract", "")))
            arxiv_id = chunk.get("arxiv_id", "unknown")

            # Format: [1. arXiv: 2301.0001]
            prompt += f"[{i}. arXiv:{arxiv_id}]\n"
            prompt += f"{chunk_text}\n\n"

        prompt += f"### Question:\n{query}\n\n"
        prompt += (
            "### Answer:\n"
            "Provide a clear, informative response based on the paper excerpts above. "
            "Cite sources using [arXiv:id] format when referencing specific information.\n\n"
        )

        return prompt
    
    def create_structured_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a prompt for Ollama with structured JSON output format.

        Uses Ollama's `format` parameter constrain output to match
        the RAGResponse schema. This ensures parseable, consistent responses.

        Args:
            query: User's question
            chunks: List of retrieved chunks

        Returns:
            Dictionary with:
                - prompt: The formatted prompt text
                - format: JSON schema for Ollama's structured output
        """
        prompt_text = self.create_rag_prompt(query, chunks)

        # Add explicit JSON formatting instructions
        prompt_text += (
            "\nRespond with a JSON object containing:\n"
            '- "answer": Your response text\n'
            '- "sources": List of arXiv PDF URLs cited\n'
            '- "confidence": "high", "medium", or "low"\n'
            '- "citations": List of arXiv IDs referenced\n'
        )

        return {
            "prompt": prompt_text,
            "format": RAGResponse.model_json_schema(),
        }
    
class ResponseParser:
    """Parser for LLM response, handling both structured and plain text.

    When structured output works, we get clean JSON. But LLMs sometimes
    ignore format instructions, so we need fallback parsing strategies.
    """

    @staticmethod
    def parse_structured_response(response: str) -> Dict[str, Any]:
        """Parse a structured JSON response from Ollama.

        Attempts parsing in order:
        1. Direct JSON parse + Pydantic validation
        2. Extract JSON from mixed text + validate
        3. Return plain text as answer with empty metadata

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary matching RAGResponse schema
        """
        # Try 1: Direct JSON parse
        try:
            parsed_json = json.loads(response)
            validated_response = RAGResponse(**parsed_json)
            logger.debug("Successfully parsed structured response")
            return validated_response.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            logger.debug(f"Direct JSON parse failed: {e}")

        # Try 2: Extract JSON from text
        return ResponseParser._extract_json_fallback(response)
    
    @staticmethod
    def _extract_json_fallback(response: str) -> Dict[str, Any]:
        """Extract JSON from response text as fallback.
        
        Sometimes LLMs wrap JSON in markdown code blocks or add
        explanatory text. This tries to find and parse the JSON portion.

        Args:
            response: Raw response text
        
        Returns:
            Dictionay with extracted content or plain text fallback
        """
        # Try to find JSON objet in the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)

        if json_match:
            try:
                parsed = json.loads(json_match.group())
                validated = RAGResponse(**parsed)
                logger.debug("Extracted JSON from mixed response")
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError) as e:
                logger.debug(f"JSON extraction failed: {e}")

        # Final fallback: treat entire rsponses as plain text answer
        logger.debug("Using plain text fallback")
        return {
            "answer": response.strip(),
            "sources": [],
            "confidence": "low",
            "citations": [],
        }
    
    @staticmethod
    def extract_citations_from_text(text: str) -> List[str]:
        """Extract arXiv IDs from citation patterns in text.
        
        Looks for patterns like:
        - [arXiv:230.00001]
        - arXiv:2301.00001                                                                                                       
        - arxiv.org/abs/2301.00001

        Args:
            text: Text potentially containing arXiv citations

        Returns:
            List of unique arXiv IDs found
        """
        patterns = [
            r'\[arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)\]', # [arXiv:2301.00001]
            r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',       # arXiv:2301.00001                                                         
            r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)',  # arxiv.org/abs/2301.00001 
        ]

        citations = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.update(matches)
        return list(citations)
