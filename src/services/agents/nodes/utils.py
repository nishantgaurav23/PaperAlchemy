"""
What is needed:
    A new file with pure utility functions shared across all 6 node files. These functions handle thje mechanical
    work of reading from state["messages"] - extracting the latest query, retrieving tool results, parsing
    SourceItems, and building reasoning steps.

Why it is neded:
    Every node needs to do the same message-inspection operations:
    - guardrail_nodes needs the latest user query to evaluate it
    - grade_document_node needs the retrieved document text to grade it
    - generate_answer_node needs both the query and the retrieved text
    - rewrite_query_node needs the query to rephrase it

Without utils.py each node would contain its own copy of for msg in reversed(messages): if isinstance(msg, HumanMessage)
... That's 4+ copies of the same logic, all of which would need to be updated if the message format ever changes.

The proper implementation needs to parse LangChain 's serialized from ToolMessage content to reconstruct SourceItem objects
from the API response.

How it helps:
# Instead of this in every node:
for msg in reversed(state["messages"]):
    if isinstance(msg, HumanMessage):
        query = msg.content
        break

# Every node just does:
from .utils import get_latest_query
query = get_latest_query(state["messages"])

The extract_sources_from_tool_messages function specifically enables the final API response to include structured source
metadata (arxiv_id, title, authors, url) rather than raw JSON strings.

____________________________________________________________________________________________________________

What it does: Pure helper functions shared across all LangGraph agent nodes.
Centralizes the repetitive work of reading from state["messages"] so each
node focuses on its own logic rather then message parsing.

┌──────────────────────────────────┬─────────────────────────────────────────────────────────────┐
  │            Function              │                          Purpose                            │
  ├──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ get_latest_query                 │ Scan messages in reverse for last HumanMessage content      │
  ├──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ get_latest_context               │ Scan messages in reverse for last ToolMessage content       │
  ├──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ extract_sources_from_tool_msgs   │ Parse ToolMessage content → list[SourceItem]                │
  ├──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ extract_tool_artefacts           │ Wrap all ToolMessages → list[ToolArtefact] for observability│
  ├──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ create_reasoning_step            │ Build a ReasoningStep record for the API response           │
  ├──────────────────────────────────┼─────────────────────────────────────────────────────────────┤
  │ filter_messages                  │ Strip ToolMessages — keep only Human + AI for LLM context  │
  └──────────────────────────────────┴─────────────────────────────────────────────────────────────┘

____________________________________________________________________________________________________________
What separate from nodes: 
    All 6 node files (guardrail, retrieve, grade, reqrite, generate, out_of_scope)
    need to read from state["messages"]. Centralising here means one change fixes all.
____________________________________________________________________________________________________________
What pure functions (not class methods):
    Nodes receives state as a plain dict. These helpers are stateless transforms on that dict -
    pure functions are the simplest, most testable form.
____________________________________________________________________________________________________________
How messages flow through the graph:
    HumanMessage <- initial user query
    AIMessage    <- LLM response (may contain tool_calls)
    ToolMessage  <- tool execution result (retrieve_papers output)
    AIMessage    <- final answer from generate_answer node
___________________________________________________________________________________________________________

Function-by-function summary

Function: get_latest_query
Called by: guardrail, grade, generate, rewrite
Returns: str
Key design decision: Reverse scan — gets most recent human message after rewrites
────────────────────────────────────────
Function: get_latest_context
Called by: grade, generate
Returns: str
Key design decision: Returns empty string (not raises) — grade_documents handles missing context gracefully
────────────────────────────────────────
Function: extract_sources_from_tool_messages
Called by: agentic_rag service
Returns: list[SourceItem]
Key design decision: JSON parse with fallback + dedup by arxiv_id
────────────────────────────────────────
Function: _parse_authors
Called by: extract_sources_from_tool_messages
Returns: list[str]
Key design decision: Private helper — normalises authors as list, comma-string, or missing
────────────────────────────────────────
Function: extract_tool_artefacts
Called by: agentic_rag service
Returns: list[ToolArtefact]
Key design decision: All ToolMessages, not just retrieve_papers
────────────────────────────────────────
Function: create_reasoning_step
Called by: all nodes
Returns: ReasoningStep
Key design decision: Thin wrapper applying metadata or {} default
────────────────────────────────────────
Function: filter_messages
Called by: rewrite node
Returns: list
Key design decision: Used when only conversation history (not tool results) should go to LLM

"""

import json
import logging
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from src.services.agents.models import ReasoningStep, SourceItem, ToolArtefact

logger = logging.getLogger(__name__)

def get_latest_query(messages: List[AnyMessage]) -> str:
    """Return the content of the most recent HumanMessages.
    
    What it does:
        Scan messages in reverse order (newest first) and returns
        the first HumanMessage's content string.

    Why it is needed:
        State accumulates multiple messages over the grapgh's life.
        After a rewrite, there may be TWO HumanMessages - the original
        and the rewritten version. Nodes always want the most recent one.

    How it helps:
        guardrail_node, grade_documents_node, generate_answer_node, and
        reqrite_query_node all call this to get the current active query.

    Args:
        messages: Full message list from state["messages]

    Returns:
        Content string of the latest HumanMessage

    Raises:
        ValueError: If no HumanMessage exists (should never happen in practice
        since the graph starts with a HumanMessage)
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
        
    raise ValueError("No HumanMessage found in state message")

def get_latest_context(messages: List[AnyMessage]) -> str:
    """Return the ocntent of the most recent ToolMessage.
    
    What it does:
        Scan messages in reverse order and returns the first ToolMessage's
        content string. This is the raw output from retrieve_papers().

    Wht it is needed:
        After ToolNode executes retrieve_papers(), it adds a ToolMessage
        containing the serialized list[Document]. grade_documents_node and 
        generate_answer_node both need this text to do their work.

    How it helps:'
        Provide the retrieved paper chunks as a single string that can be
        directly embedded into LLM prompts (GRADE_DOCUMENTS_PROMPT,
        GENERATE_ANSWER_PROMPT) using .format(context=...).

    Args:
        messages: Full message list from state["messages]

    Returns:
        Content string of the latest ToolMessage, or empty string if none exists
        (empty means no retrieval has happened yet - grade_documents handles this
        by routing to rewrite_query)
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    
    return ""

def extract_sources_from_tool_messages(messages: List[AnyMessage]) -> List[SourceItem]:
    """Parse ToolMessage content from retrieve_papers into SourceItem objects.
    
    What it does:
        Find ToolMessages from the "retrieve_papers" tool, parses their
        JSON content (list of serialized LangChain Documents), and converts
        each Document's metadata into a SourceItem for the API response.

    Why it is needed:
        LangGraph's ToolNode serializes the list[Dcoument] returned by
        retrieve_papers() into a JSON string in ToolMessage.content.
        The API response needs structured SourceItem objects (arxiv_id, 
        title, url) - not raw JSON strings.

    How it helps:
        AgenticRAGServices._extract_sources() calls this to build the
        `sources` field in the final API response. Without this, sources
        would be empty even when papers was retrieved.

    Args:
        messages: Full messages list from state["messages"]

    Returns:
        List of SourceItem objects, one per unique arxiv_id found across all
        retrieve_papers tool calls. Returns empty list if none found or parse fails.
    """
    sources: List[SourceItem] = []
    seen_arxiv_ids: set = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if getattr(msg, "name", None) != "retrieve_papers":
            continue

        content = msg.content
        if not content:
            continue

        try:
            # LangChain ToolNode serializes list[Documents] as JSON
            data = json.loads(content) if isinstance(content, str) else content

            if not isinstance(data, list):
                continue

            for item in data:
                # Support both dict-form and Document form
                if isinstance(item, dict):
                    metadata = item.get("metadata", {})
                    page_content = item.get("page_content", "")
                else:
                    # Fallback: try attribute access
                    metadata = getattr(item, "metadata", {})
                    page_content = getattr(item, "page_content", "")

                arxiv_id = metadata.get("arxiv_id")
                if not arxiv_id or arxiv_id in seen_arxiv_ids:
                    continue

                seen_arxiv_ids.add(arxiv_id)
                sources.append(
                    SourceItem(
                        arxiv_id=arxiv_id,
                        title=metadata.get("title", ""),
                        authors=_parse_authors(metadata.get("authors", [])),
                        url=metadata.get("source", f"https://arxiv.org/pdf/{arxiv_id}.pdf"),
                        relevance_score=float(metadata.get("score", 0.0))
                    )
                )

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(f"Could not parse ToolMessage content as source: {e}")
            continue
    
    logger.debug(f"Extracted {len(sources)} unique sources from tool messages")
    return sources

def _parse_authors(raw: object) -> List[str]:
    """Normalize the authors field from OpenSearch hits.
    
    What it does:
        OpenSearch may stire authors as a list, a comma-separated string,
        or missing entirely. This returns a consistent List[str].


    Args:
        raw: The raw authors value from document metadata

    Returns: List of author name strings
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        return [a.strip() for a in raw.split(",") if a.strip()]
    return []

def extract_tool_artefacts(messages: List[AnyMessage]) -> List[ToolArtefact]:
    """Wrap all ToolMessages into ToolArtefat objects for observablity.
    
    What it does:
        Iterates all messages, finds every ToolMessage regardless of tool name,
        and wraps each into a ToolArtefact that records what was called and what it returned.

    Why it is needed:
        Langfuse tracing and the reasoning steps in the API response need to show
        WHICH tools were called , with what IDs, and what they returned.
        ToolArtefact provides a typed, consistent record of this.

    How it helps:
        AgenticRAGServie._extract_reasoning_steps() calls this to build
        the observability trail shown to users and in LangFuse.

    Args:
        messages: Full message list from state["messages]

    Returns:
        List of ToolArtefacts objects, one per ToolMessage found
    """
    artefacts: List[ToolArtefact] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        artefacts.append(
            ToolArtefact(
                tool_name=getattr(msg, "name", "unknown"),
                tool_call_id=getattr(msg, "tool_call_id", ""),
                content=msg.content,
                metadata={},
            )
        )
    return artefacts

def create_reasoning_step(
        step_name: str,
        description: str,
        metadata: Optional[Dict] = None,
) -> ReasoningStep:
    """Build a ReasoningStep record for inclusion in the api response.
    What it does:
        Constructs a ReasoningStep Pydantic model with the given name,
        human-readable description, and optional metadata dict.

    Wht it is needed:
        The agentic API response includes a `reasoning_steps` fields so users
        can see exactly what decisons the agent made. Each node calls this to contribute one step to
        the trace.

    How it helps:
        Typing  `ReasoningStep(step_name=..., description=..., metadata=...)`
        everytime is verbose and error-prone. This wrapper applies the `metadata or {}`
        defaults and ensure consistent constructions.


    Args:
        step_name:  Short identifer for the step (e.g., "guardrail", "grading")
        description: Human-readable explaination )e.g., "Query scored 85/100")
        metadata: Optional dict of structured data (scores, counts, timings)

    Retruns:
        A reasoningstep ready to append to the reasoning_steps list
    """
    return ReasoningStep(
        step_name=step_name,
        description=description,
        metadata=metadata or {},
    )

def filter_messages(messages: List[AnyMessage]) -> List[AIMessage | HumanMessage]:
    """Strip ToolMessages, keeping only Human and AI messages.
    What it does:
        Returns a new list containing only HumanMessage and AIMessage
        instances. ToolMessages (raw retrieval output) are excluded.

    Why it is needed:
        Some LLM calls (like rewrite_query) should see the conversation
        history but NOT the raw document chunks from ToolMessages - those
        are noise that inflates the context window and confuses the reqrite.

    How it helps:
        Pass the filtered list into LLM prompts to get a cleaner, more
        focussed conversation history without retrieval artifacts.

    Args:
        messages: Full message list from state["messages"]

    Returns:
        New list with only HumanMessage and AIMessage instance
    """
    return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]
    