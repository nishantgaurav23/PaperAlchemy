"""
What is needed:
    A terminal node that handles queries the guardrail rejected (score below threshold). It returns a polite, informative
    rejection message explaining PaperAlchemy's domain scope.

Why it is needed:
    When continue_after_guardrail returns "out_of_scope", LangGraph routes here. The graph must end with a response — it can't
    just silently stop. This node provides a structured AIMessage that AgenticRAGService._extract_answer() can read and return
    to the user.

    Without this node, rejected queries would hit a dead end in the graph and raise an error.

How it helps

  "What is 2+2?" → guardrail scores 5/100 → out_of_scope
      → ainvoke_out_of_scope_step adds AIMessage to state["messages"]
      → graph ends
      → service reads last AIMessage → returns to user:
          "I can only help with CS/AI/ML research papers..."

Why no LLM call here:
    prompts.py defines DIRECT_RESPONSE_PROMPT with a note "Used by: out_of_scope_node". However, making an LLM
    call in a rejection node is deliberately avoided.

      ┌─────────────┬───────────────────────────────────────────────────────────────────┐
  │   Reason    │                              Detail                               │
  ├─────────────┼───────────────────────────────────────────────────────────────────┤
  │ Speed       │ Hardcoded response is instant. No LLM round-trip for a rejection. │
  ├─────────────┼───────────────────────────────────────────────────────────────────┤
  │ Reliability │ If Ollama is down, rejection still works. LLM call would fail.    │
  ├─────────────┼───────────────────────────────────────────────────────────────────┤
  │ Consistency │ Users always see the same clear boundary statement.               │
  ├─────────────┼───────────────────────────────────────────────────────────────────┤
  │ Cost        │ No tokens consumed on off-domain queries.                         │
  └─────────────┴───────────────────────────────────────────────────────────────────┘

  DIRECT_RESPONSE_PROMPT is available if you ever want to switch to a generated response — just swap the hardcoded string for
  an LLM call with that prompt.

__________________________________________________________________________________________________
What it does:
    Terminal node for queries that failed guardrail validation. Returns a 
    hardcoded rejection AIMessage explaining the PaperAlchemy's domain scope, 
    then the graph ends.

  ┌──────────────────────────────┬───────────────────────────────────────────────────────────────┐
  │           Symbol             │                           Purpose                             │
  ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ ainvoke_out_of_scope_step    │ Async node — appends rejection AIMessage to state["messages"] │
  └──────────────────────────────┴───────────────────────────────────────────────────────────────┘

__________________________________________________________________________________________________
Why no LLM call:
    Rejection messages should be instant, consistent, and never fail. Making an LLM
    call here would add latency, consume tokens, and risk another failure point for
    queries that were already rejected. A hardcoded response achieves the same result
    faster and more reliably.

    prompts.py defines DIRECT_RESPONSE_PROMPT for this node - it is available if you later
    want to seitch to a dynamically generated rejection.

Why AIMessage (not HumanMessage):
    The graph represents responses from the assistant as AIMessage. The service's 
    _extract_answer() methoda reads the last AIMessage from state['messages] as the
    final answer. ToolMessages and HumanMessages are skipped.

Why return {"messages": [...]}:
    The add_message reducer on state["messages] APPENDS this AIMessage to the
    existing message list (HumanMessage + any prior messages). The graph then
    ends at the END node and the service read state["messages"][-1] for the
    answer.
__________________________________________________________________________________________________

Execution trace

state["messages"] = [HumanMessage(content="What is 2+2?")]
                    (guardrail scored 5, threshold 40 → out_of_scope)

Step 1: get_latest_query → "What is 2+2?"
Step 2: build response_text with question embedded
Step 3: return {"messages": [AIMessage(content="I can only help...")]}

LangGraph add_messages reducer appends →
state["messages"] = [
    HumanMessage("What is 2+2?"),
    AIMessage("I can only help with questions about academic research...")
]

Graph routes → END

AgenticRAGService._extract_answer() reads last AIMessage → returns to user

Why runtime is in the signature even though it's unused

LangGraph requires all node functions to have the same signature: (state, runtime). Even nodes that don't need the runtime
context must declare it. Removing it would cause a signature mismatch at graph-build time.
"""

import logging
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.services.agents.context import Context
from src.services.agents.state import AgentState
from src.services.agents.nodes.utils import get_latest_query

logger = logging.getLogger(__name__)

async def ainvoke_out_of_scope_step(
        state: AgentState,
        runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Terminal node: return a rejection messages for out-of-scope queries.
    
    What it does:
        Reads the rejected query from state, constructs a clear rejection
        message that explains PaperAlchemy's domain scope, and returns it
        as an AIMessage. The graph routes here -> END with no further nodes.

    Why it is needed:
        LangGraph graphs must produce a terminal state. When guardrail rejects
        a query, the graph still needs to write a response to state["messages"]
        so the service can extract and return it to the user.

    How it helps:
        - No LLM call → instant response, zero Ollama dependency
        - Hardcoded text → consistent user experience, easy to update in one place
        - Includes the rejected query → user can see what was rejected
        - Suggests alternatives → constructive rejection, not a dead end

    Args:
        state: Current agent state (reads state["messages"] for the query)
        runtime: Runtime context (not used — no LLM call, no Langfuse span needed)

    Returns:
        Partial state dict: {"messages": [AIMessage(content=rejection_text)]}
        The add_messages reducer appends this to state["messages"].
    """

    logger.info("NODE: out_of_scope")

    question = get_latest_query(state["messages"])
    logger.info(f"Rejecting out-of-scope query: {question[:100]}...")

    response_text = (
        "I can only help with questions about academic research papers "
        "in Computer Science, Artificial Intelligence, and Machine Learning from arXiv.\n\n"
        f"Your question: '{question}'\n\n"
        "This appears to be outside my domain of expertise. For questions like this, you might want to try:\n"
        "- General-purpose AI assistants for broad knowledge questions\n"
        "- Domain-specific resources for topics outside CS/AI/ML\n"
        "- Technical documentation if asking about specific software or tools\n\n"
        "If you have a question about AI/ML/CS research papers, I am happy to help!"
    )

    return {"messages": [AIMessage(content=response_text)]}