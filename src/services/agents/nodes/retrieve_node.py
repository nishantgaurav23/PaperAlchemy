"""
What is needed:
    A node that either creates an AIMessage with tool_calls (triggering LangGraph's ToolNode to execute retrieve_papers)
    or return a hardcoded fallback message if the maximum retrieval attempts has been reached.

Why it is needed:
    This node is the trigger for retrieval, not the retriever itself. The actual embedding + OpenSearch happens
    inside the @tool function (tools.py). This node's job is to create a properly-formed AIMessage with tool_calls
    that tells ToolNode what to call.

Why this separation matters:
    - ToolNode is LangGraph's build-in node that executes @tool functions. It reads AIMessage.tool_calls that tells 
      ToolNode what to call.

    - retrieve_node can't call that tool directly - it must go through ToolNode so LangGraph handles the serialization
      and message routing correctly.
    - The loop guard (max attempts) lives here, not in the tool - it's a workflow-level concern, not a retrieval
      concern.

How it helps:

Attempt 1: retrieve_node creates AIMessage(tool_calls={retrieve_papers, query}])
    -> ToolNode executes retrieve_papers(query) -> adds ToolMessage to state
    -> grade_documents grades results -> "irrelevant" -> rewrite_query
    -> rewrite_query imporved query -> loop back to retrive_node

Attempt 2: same flow with rewritten query

Attempt 3 (= max_attempts): retrieve_node detects current_attempts >= max_attempts
    -> returns AIMessage(fallback text, no tool_calls)
    -> ToolNode passes through (no tool_calls to execute)
    -> grade_documents routes to generate_answer (empty context -> answer with what's available)

What it does: Triggers document retrieval by creating an AIMessage with tool_calls that
  LangGraph's ToolNode executes. Acts as the loop controller — tracks retrieval attempts
  and returns a hardcoded fallback when the maximum is reached.

  ┌─────────────────────────────┬──────────────────────────────────────────────────────────────────┐
  │           Symbol            │                             Purpose                              │
  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ ainvoke_retrieve_step       │ Async node — creates tool_calls AIMessage OR returns fallback    │
  │                             │ when max_retrieval_attempts is reached                           │
  └─────────────────────────────┴──────────────────────────────────────────────────────────────────┘

Why this node does NOT call the retrieval tool directly:
    LangGraph's ToolNode is the canonical way to execute @tool functions. It:
    1. Reads AIMessage.tool_calls from the last message in state
    2. Calls the matching @tool function
    3. Wraps the result in a ToolMessage and appends it to state.messages
    This separation lets ToolNode handle serialization, error catching, and
    message routing — we get this for free by following the protocol.

Why an empty AIMessage content (""):
    When a node returns an AIMessage(content="", tool_calls=[...]), the LLM is
    saying "I have nothing to say, but I need to call a tool." This is the
    standard LangGraph tool-calling pattern. ToolNode keys on the presence of
    tool_calls, not the content string.

Why track retrieval_attempts here (not in the tool):
    retrieval_attempts is a workflow-level counter — it governs how many times
    the retrieve → grade → rewrite loop runs. The @tool function is stateless
    and doesn't know about the loop. The retrieve node is the loop entry point,
    so it is the right place to increment and check this counter.

Why original_query is saved only on attempt 1:
    After rewrite_query_node runs, state["messages"] contains a new HumanMessage
    with the rewritten query. get_latest_query() would return the rewritten version.
    But we need the original query for tracing, cache keys, and the API response.
    We save it only once (when original_query is None) so it is never overwritten.

"""

import logging
import time
from typing import Dict, Union

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.services.agents.context import Context
from src.services.agents.state import AgentState
from src.services.agents.nodes.utils import get_latest_query

logger = logging.getLogger(__name__)

async def ainvoke_retrieve_step(
        state: AgentState,
        runtime: Runtime[Context],
) -> Dict[str, Union[int, str, list]]:
    """Async node: create a retrieval tool call or return a fallbacl when attempts are exhausted.
    
    What it does:
        1. Reads the current query (possibly rewritten) from state.messages
        2. On first call: saves original_query to state
        3. Checks retriveal_attempts against max_retrieval_attempts
        4. If under limit: increments counter and creates tool_calls AIMessage
        5. If at limit: returns a hardcoded fallback AIMessage (no tool calls)
        6. Optionally records a Langfuse span

    Why it is needed:
        Without this node, the graph has no way to. trigger ToolNode. ToolNode
        only runs when the lats message in state contains tool_calls - this
        node is responsible for creating that message.

    How it helps:
        - Creates the exact AIMessage format ToolNode expects: content="" + 
          tool_cals=[{id, name, args}]
        - The tool_call id includes the attempt number for traceability
        - original_query is saved once so tracing always shows the user's 
          real intent, not the rewrittem version
        - Fallback path prevents infinite loops without crashing

    Args:
        state: Current agent state (reads_messages, retrieval_attempts, orignal query)
        runtime: Runtime context (read max_retrieval_attempts, top_k, langfuse config)

    Returns:
        Partial state dict containing:
        - retrieval_attempts: increment attempt count
        - original_query: set on first call only
        - messages: [AIMessage(tool_calls=[retrieve_papers])] or [AIMessage(fallback)]
    """
    logger.info("NODE: retrieve")
    start_time =  time.time()

    messages = state["messages"]
    question = get_latest_query(messages)
    current_attempts = state.get("retrieval_attempts", 0)
    max_attempts = runtime.context.max_retrieval_attempts

    # --------Preserve original query (before any rewrites) ------------
    # Only set on the first call. After rewrite_query_node runs, messages
    # contain a new HumanMessage with the rewritten text - get_latest_query
    # would return the rewrite, not the original.
    updates: Dict[str, Union[int, str, list]] = {}
    if state.get("original_query") is None:
        updates["original_query"] = question
        logger.debug(f"Saved original query: {question[:100]}...")

    # --- Start Langfuse span----------------------------------------------
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.start_span(
                trace=runtime.context.trace,
                name="document_retrieval_initiation",
                metadata={
                    "node": "retrieve",
                    "top_k": runtime.context.top_k,
                    "attempt": current_attempts + 1,
                    "max_attempts": max_attempts,
                },
                input={"query": question},
            )
        except Exception as e:
            logger.warning(f" FAiled to create Langfuse span for retrieve node: {e}")

    # ---- Max attempts fallback ---------------------------------------------------
    # When the retrive -> grade -> rewrite loop has run max_attempts times,
    # stop looping and eturn a graceful degradation message. The graph
    # then routes through ToolNode (no-op) -> grade_documents -> generate_answer.
    if current_attempts >= max_attempts:
        logger.warning(f"Max retrieval attempts ({max_attempts}) reached - returning fallback")


        fallback_msg = (
            f"I was unable to find relevant research papers after {max_attempts} attempts.\n"
            "This may be because:\n"
            "1. No papers in the database contain information about this topic\n"
            "2. The query terms do not match the indexed content\n\n"
            "Please try rephrasing your question with more specific technical terms "
            "such as model names, algorithm names, or paper titles."
        )

        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"status": "max_attempts_reached", "fallback": True},
                metadata={"execution_time_ms": execution_time},
            )

        return {**updates, "messages": [AIMessage(content=fallback_msg)]}
    
    # --------Normal path: create tool call ----------------------
    # Increment counter before creating the tool call so the next node
    # (after ToolNode) can read the updated count.
    new_attempt_count = current_attempts + 1
    updates["retrieval_attempts"] = new_attempt_count
    logger.info(f"Creating retrieval tool call: attempts {new_attempt_count} / {max_attempts}")

    # AIMessage with tool_calls tells ToolNode to execute retrieve_papers(query).
    # - content="" is required (LangChain validates it's a string, even if empty)
    # - id must be unique — using attempt number ensures traceability in Langfuse
    # - name must match the @tool function name exactly ("retrieve_papers")
    # - args must match the @tool's parameter names exactly ({"query": ...})
    updates["messages"] = [
        AIMessage(
            content="",
            tool_calls=[ 
                {
                    "id": f"retrieve_{new_attempt_count}",
                    "name": "retrieve_papers",
                    "args": {"query": question},
                }
            ],
        )
    ]

    logger.debug(f"Tool call created for: {question[:100]}...")

    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.update_span(
            span,
            output={
                "status": "tool_call_created",
                "query": question,
                "attempt": new_attempt_count,
            },
            metadata={"execution_time_ms": execution_time},
        )
    return updates
