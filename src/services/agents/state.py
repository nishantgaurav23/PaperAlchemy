"""
What it does: Defines the `AgenticState` TypeDict -the single data structure that flows through every node in the LangGraph
workflow. Each node reads from and writes to this shared state.

Field: messages
Type: list
Purpose: LangChain message history (HumanMessage, AIMessage, ToolMessage) - LangGraph's core communication channel
-----------------------------------------
Field: original_query
Type: str
Purpose: Preserved original query (unchaned even after rewriting)
-----------------------------------------
Field: rewritten_query
Type: str
Purpose: Improved query after rewrite node (None if no rewrite needed)
-----------------------------------------
Field: guardrail_result
Type: GuardrailScoring
Purpose: Score + reason from domain validation
-----------------------------------------
Filed: routing_decision
Type: str
Purpose: Current routing target ("generate_answer" or "rewrite_query)
-----------------------------------------
Field: sources
Type: list
Purpose: Raw searh results from OpenSearch
-----------------------------------------
Field: relevant_sources
Type: list[SourceItem]
Purpose: Filtered sources that passed grading
-----------------------------------------
Field: relevant_tool_artefacts
Type: list[ToolArtefact]
Purpose: Tool call metadata or observability
-----------------------------------------
Field: grading_results
Type: list[GradingResults]
Purpose: Per-document grading outcomes
-----------------------------------------
Field: metadata:
Type: dict
Purpose: Extensible metadata bucket (timing. model, info, etc.)
Why TypeDict over dataclass: LangGraph requires state to be a TypeDict (or Annotated dict) because it merges partial
updates from each noode. Dataclasses don't support this merge-on-return pattern.

LangGraph agent state definition.

This typeDict is the single source of truth that flows through every node
in the agentic RAG workflow. LangGraph's execution model works like this:
    1. Graph starts with initial state dict
    2. Each node receives the FULL State
    3. Each node returns a PARTIAL dict of only the fields it changed
    4. LangGraph merges the partial update into the full state.
    5. Next node receives the updated full state

This merge-on-return pattern is why we use TypeDict (not dataclass/Pydantic):
- TypeDict allows partial returns (just the keys you changed)
- LangGraph handles the merge automatically.
- Type hints still give IDE autocompletion

Field design principles:
- `messages`: LangGraph's built-in message passing. ToolNode reads/writes
   tool calls and tool results here automatically.
- `retrieval_attempts`: Prevents infinite loops. The grade -> rewrite -> retrieve
   cycle increments this counter; retrieve_node checks it against max-attempts.
- `relevant_sources` vs `sources`: sources holds raw OpenSearch hits;
   relevant_sources holds only those that passed the grading node. tHE
   generate node uses relevant_sources for answer context.
- `metadata`: Intentionally unstructured. Nodes can stash. timing data,
   model info, or debug data here without changing the state schema.
"""

from typing import Any, Dict, List, Optional

from typing_extensions import TypeDict

from src.services.agents.models import (
    GradingResult,
    GuardrailScoring,
    SourceItem,
    ToolArtefact,
)

class AgentState(TypeDict, total=False):
    """State flowing through the agentic RAG LangGraph workflow.
    Every node reads from this state and returns a partial dict
    contianing only the fields it modified. LangGraph merges the
    partial updates automatically.

    Flow:
        guardrail → retrieve → tool_retrieve → grade_documents
              ↓ (if irrelevant)      ↓ (if relevant)
        rewrite_query          generate_answer
            ↓
        retrieve (retry)
    total=False means all the fields are optional in partial returns.
    The initial state in agentic_rag.py sets all required defaults.
    """

    # -- Core message history --------------------------------------
    # LangGraph's primary communication channel between nodes.
    # HumanMessage (user query), AIMessage (LLM response + tool calls),
    # ToolMessage (retriever results). ToolNode reads/writes here.
    messages: List[Any]

    # -- Query tracking --------------------------------------
    # origianl_query: Never modified - used for tracing and cache keys
    # rewritten_query: Set by re-write_query_node when grading fails
    original_query: Optional[str]
    rewritten_query: Optional[str]

    # -- Retrieval control--------------------------------------
    # Incremented each retrieve attempt. Checked against
    # context.max_retrieval_attempt to prevent infinite loops.
    retrieval_attempts: int

    # -- Guardrail--------------------------------------
    # Set by guardrail_node. The conditional edge reads .score to decide
    # whether to route to retrieve (score >= threshold) or out_of_scope.
    guardrail_result: Optional[GuardrailScoring]

    # ── Routing ─────────────────────────────────────────────────────────
    # Set by grade_documents_node. Read by the conditional edge after
    # grading: "generate_answer" if relevant, "rewrite_query" if not.
    routing_decision: Optional[str]

    # ── Search results ──────────────────────────────────────────────────
    # sources: Raw hits from OpenSearch (set by tool_retrieve via ToolNode)
    # relevant_sources: Filtered by grade_documents_node (only relevant ones)
    source: Optional[List[Dict[str, Any]]]
    relevant_sources: List[SourceItem]

    # ── Tool metadata ───────────────────────────────────────────────────
    # Captures tool_call IDs and results for LangFuse observability
    relevant_tool_artefacts: Optional[List[ToolArtefact]]

    # ── Grading outcomes ────────────────────────────────────────────────
    # Per document grading results. Used to build reasoning steps
    # and to count relevant vs irrelevant documents.
    grading_Results: List[GradingResult]

    # ── Extensible metadata ─────────────────────────────────────────────
    # Nodes can stash any data here (timing, model, info, debug).
    # Not validated - intentionally flexible
    metadata: Dict[str, Any]
