"""
What it does: Defines the `AgentState` TypedDict — the single data structure that flows through every node in the LangGraph
workflow. Each node reads from and writes to this shared state.

Field: messages
Type: Annotated[list[AnyMessage], add_messages]
Purpose: LangChain message history (HumanMessage, AIMessage, ToolMessage) — LangGraph's core communication channel.
        The add_messages reducer APPENDS new messages instead of overwriting, preserving conversation history.
-----------------------------------------
Field: original_query
Type: str
Purpose: Preserved original query (unchanged even after rewriting)
-----------------------------------------
Field: rewritten_query
Type: str
Purpose: Improved query after rewrite node (None if no rewrite needed)
-----------------------------------------
Field: guardrail_result
Type: GuardrailScoring
Purpose: Score + reason from domain validation
-----------------------------------------
Field: routing_decision
Type: str
Purpose: Current routing target ("generate_answer" or "rewrite_query")
-----------------------------------------
Field: sources
Type: dict
Purpose: Raw search results from OpenSearch (keyed by tool_call_id)
-----------------------------------------
Field: relevant_sources
Type: list[SourceItem]
Purpose: Filtered sources that passed grading
-----------------------------------------
Field: relevant_tool_artefacts
Type: list[ToolArtefact]
Purpose: Tool call metadata for observability
-----------------------------------------
Field: grading_results
Type: list[GradingResult]
Purpose: Per-document grading outcomes
-----------------------------------------
Field: metadata
Type: dict
Purpose: Extensible metadata bucket (timing, model info, etc.)

Why TypedDict over dataclass: LangGraph requires state to be a TypedDict (or Annotated dict) because it merges partial
updates from each node. Dataclasses don't support this merge-on-return pattern.

LangGraph agent state definition.

This TypedDict is the single source of truth that flows through every node
in the agentic RAG workflow. LangGraph's execution model works like this:
    1. Graph starts with initial state dict
    2. Each node receives the FULL state
    3. Each node returns a PARTIAL dict of only the fields it changed
    4. LangGraph merges the partial update into the full state
    5. Next node receives the updated full state

This merge-on-return pattern is why we use TypedDict (not dataclass/Pydantic):
- TypedDict allows partial returns (just the keys you changed)
- LangGraph handles the merge automatically
- Type hints still give IDE autocompletion

Field design principles:
- `messages`: Uses Annotated[..., add_messages] reducer so LangGraph APPENDS
    new messages instead of overwriting. Without this, each node would destroy
    the conversation history. ToolNode reads/writes tool calls and results here.
- `retrieval_attempts`: Prevents infinite loops. The grade -> rewrite -> retrieve
    cycle increments this counter; retrieve_node checks it against max_attempts.
- `relevant_sources` vs `sources`: sources holds raw OpenSearch hits;
    relevant_sources holds only those that passed the grading node. The
    generate node uses relevant_sources for answer context.
- `metadata`: Intentionally unstructured. Nodes can stash timing data,
    model info, or debug data here without changing the state schema.
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.services.agents.models import (
    GradingResult,
    GuardrailScoring,
    SourceItem,
    ToolArtefact,
)


class AgentState(TypedDict, total=False):
    """State flowing through the agentic RAG LangGraph workflow.

    Every node reads from this state and returns a partial dict
    containing only the fields it modified. LangGraph merges the
    partial updates automatically.

    Flow:
        guardrail → retrieve → tool_retrieve → grade_documents
            ↓ (if irrelevant)      ↓ (if relevant)
        rewrite_query          generate_answer
            ↓
        retrieve (retry)

    total=False means all fields are optional in partial returns.
    The initial state in agentic_rag.py sets all required defaults.

    Critical: The `messages` field uses the add_messages reducer.
    This tells LangGraph to APPEND new messages rather than replace
    the list. Without this, every node would overwrite the entire
    conversation history.
    """

    # ── Core message history ──────────────────────────────────────────
    # LangGraph's primary communication channel between nodes.
    # HumanMessage (user query), AIMessage (LLM response + tool calls),
    # ToolMessage (retriever results). ToolNode reads/writes here.
    #
    # The add_messages reducer is CRITICAL:
    #   - Without it: node returns {"messages": [X]} → state.messages = [X] (overwrite)
    #   - With it:    node returns {"messages": [X]} → state.messages = [...existing, X] (append)
    messages: Annotated[list[AnyMessage], add_messages]

    # ── Query tracking ────────────────────────────────────────────────
    # original_query: Never modified — used for tracing and cache keys
    # rewritten_query: Set by rewrite_query_node when grading fails
    original_query: Optional[str]
    rewritten_query: Optional[str]

    # ── Retrieval control ─────────────────────────────────────────────
    # Incremented each retrieve attempt. Checked against
    # context.max_retrieval_attempts to prevent infinite loops.
    retrieval_attempts: int

    # ── Guardrail ─────────────────────────────────────────────────────
    # Set by guardrail_node. The conditional edge reads .score to decide
    # whether to route to retrieve (score >= threshold) or out_of_scope.
    guardrail_result: Optional[GuardrailScoring]

    # ── Routing ───────────────────────────────────────────────────────
    # Set by grade_documents_node. Read by the conditional edge after
    # grading: "generate_answer" if relevant, "rewrite_query" if not.
    routing_decision: Optional[str]

    # ── Search results ────────────────────────────────────────────────
    # sources: Raw hits from OpenSearch (set by tool_retrieve via ToolNode)
    # relevant_sources: Filtered by grade_documents_node (only relevant ones)
    sources: Optional[Dict[str, Any]]
    relevant_sources: List[SourceItem]

    # ── Tool metadata ─────────────────────────────────────────────────
    # Captures tool_call IDs and results for Langfuse observability
    relevant_tool_artefacts: Optional[List[ToolArtefact]]

    # ── Grading outcomes ──────────────────────────────────────────────
    # Per-document grading results. Used to build reasoning steps
    # and to count relevant vs irrelevant documents.
    grading_results: List[GradingResult]

    # ── Extensible metadata ───────────────────────────────────────────
    # Nodes can stash any data here (timing, model info, debug).
    # Not validated — intentionally flexible.
    metadata: Dict[str, Any]