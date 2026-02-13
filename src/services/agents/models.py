"""
File: src/services/agents/models.py                                                                                         
Purpose: Pydantic models for agent state (scoring, grading, routing)

Pydantic models for the agentic RAG workflow

These models serves two purposes:
1. Structured LLM output - GuardrailScoring, GradeDocuments are used with
Ollams's JSON format to get reliable, parseable responses from the LLM
instead of free-form text,
2. State tracking - SourceItem, GuardingResult, ReasoningStep track what
happened at each stage so the final response can include full reasoning
transparency.

Why pydantic over plain dicts:
- Validation at parse time catches malformed LLM outputs immediately
- Filed constraints (ge=0, le=100, Literal["yes", "no"]) prevent silent
- .to_dict() on SourceItem gives clean JSON for API responses

What it does: 
- Defines all Pydantic models used across agent nodes for type-safe structured LLM outputs and state management.
  ┌──────────────────┬───────────────────────────────────────────────────────┐
  │      Model       │                        Purpose                        │
  ├──────────────────┼───────────────────────────────────────────────────────┤
  │ GuardrailScoring │ LLM output: relevance score (0-100) + reason          │
  ├──────────────────┼───────────────────────────────────────────────────────┤
  │ GradeDocuments   │ LLM output: binary relevance ("yes"/"no") + reasoning │
  ├──────────────────┼───────────────────────────────────────────────────────┤
  │ SourceItem       │ Structured source reference (arxiv_id, title, url)    │
  ├──────────────────┼───────────────────────────────────────────────────────┤
  │ ToolArtefact     │ Metadata wrapper for tool call results                │
  ├──────────────────┼───────────────────────────────────────────────────────┤
  │ RoutingDecision  │ Next-node routing with reason                         │
  ├──────────────────┼───────────────────────────────────────────────────────┤
  │ GradingResult    │ Per-document grading outcome                          │
  ├──────────────────┼───────────────────────────────────────────────────────┤
  │ ReasoningStep    │ Human-readable reasoning trace for transparency       │
  └──────────────────┴───────────────────────────────────────────────────────┘
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

class GuardrailScoring(BaseModel):
    """LLM output for guardrail validation.
    The guardrail node asks the LLM to sore whether a query falls within
    the CS/AI/ML reserach domain. Score below the threshold route to
    the out_of_scope node instead of retrieval.

    Score ranges:
        80-100: Clearly CS/AI/ML research
        60-79: Potentially research-related
        40-59: Borderline / ambiguous
        0-39: Not about reserach papers
    """
    score: int = Field(ge=0, le=100, description="Relevance score between 0 and 100")
    reason: str = Field(description="Brief reason for the score")

class GradeDocuments(BaseModel):
    """LLM output for document relevance grading.
    
    After retrieval, the grading node asks the LLM whether retrieved
    documents are relevant to the query. Use binary scoring for
    clear routing decisions (generate vs rewrite).
    """
    binary_score: Literal["yes", "no"] = Field(
        description= "Document relevance: 'yes' or 'no'")
    reasoning: str = Field(default="", description="Explanation for the decision")

class SourceItem(BaseModel):
    """Structured source reference from retrieved documents.

    Extracted from OpenSearch hits and filtered by the grading node.
    Only relevant source appear in the final API response.
    """

    arxiv_id: str = Field(description="arXiv paper ID")
    title: str = Field(description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    url: str = Field(description="Link to paper")
    relevance_score: float = Field(default=0.0, description="Relevance sore from search")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "relevance_score": self.relevance_score,
        }
    
class ToolArtefact(BaseModel):
    """Metadata wrapper for tool call results.
    
    Captures what each tool returned so the final reponse can
    include retrieval metadata (which tool, how many results, etc.).
    """

    tool_name: str = Field(description="Name of the tool")
    tool_call_id: str = Field(description="Unique tool call ID")
    content: Any = Field(description="Tool result content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class RoutingDecision(BaseModel):
    """Routing decision for graph navigation.
    
    Used internally by conditional edges to determine which node
    executes next based on guardrail scores or grading results.
    """

    route: Literal["retrieve", "out_of_scope", "generate_answer", "rewrite_query"] = Field(
        description="Next node to route to"
    )
    reason: str = Field(default="", description="Reason for routing decision")

class GradingResult(BaseModel):
    """Per-document grading outcome.
    
    Stored in state so the reasoning steps can report how many
    documents were graded relevant vs irrelevant.
    """

    document_id: str = Field(description="Dcoument identifier")
    is_relevant: bool = Field(description="Relevance flag")
    score: float = Field(default=0.0, description="Relevance score")
    reasoning: str = Field(default="", description="Grading reasoning")

class ReasoningStep(BaseModel):
    """Human-readable reasoning trace.
    
    Collected aross all nodes and returned in the API response
    so users (and Langfuse) can see exactly what decisions the
    agent made and why.
    """

    step_name: str = Field(description="Name of the reasoning step")
    description: str = Field(description="Human-readable description")
    metadata: Dict[str, any] = Field(default_factory=dict, description="Step metadata")