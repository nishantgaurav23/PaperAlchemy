"""
What is needed: 
    The first node in the grapgh. It receives the user's query, asks the LLM to score it on a 0-100 domain
    relevance scale using structured output, and stores the result in state. A companion conditional edge
    function (continue_after_guardrail) reads that score to decide whether the workflow continues to retrieval
    or stops at out_of_scope.

Why it is needed:
    Without domain validation, every query - "What is a dog?", "Hello", "2+2?" - would trigger a full
    retrieval + grading + generation cycle, wasting compute on guranteed-wrong results. The guardrail is
    the cheapest possible check: a single fast LLM call with temperature=0.0 (deterministic) that classifies
    the query before anything expensive happens.

    The continue_after_guardrail function is a contional edge, not a node. LangGraph calls it synchronously after
    ainvoke_guardrail_step completes and uses its return value ("continue" or "out_of_scope") to select
    teh next node. This is why it returns a Literal string, not a dict.

How it helps:
User asks: "What is a dog?"
    → guardrail scores it 10/100 (below threshold 40)
    → continue_after_guardrail returns "out_of_scope"
    → graph routes to out_of_scope node → user gets polite rejection

User asks: "How do transformers work?"
    → guardrail scores it 92/100 (above threshold 40)
    → continue_after_guardrail returns "continue"
    → graph routes to retrieve node → full RAG pipeline runs

Langfuse API : 
PaperAlchemy: langfuse_tracer.start_span(trace, name, metadata={...}, input={...})
PaperAlchemy: langfuse_tracer.update_span(span, output={...}, metadata={...}) ← also ends it
PaperAlchemy: Single update_span(span, ...) call

___________________________________________________________________________________________________________
What it does: First node in the agentic RAG graph. Uses a structured LLM call to score
the user query (0-100) for domain relevance, then routes the grapgh via conditional edge.

  ┌─────────────────────────────┬──────────────────────────────────────────────────────────────────┐
  │           Symbol            │                             Purpose                              │
  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ continue_after_guardrail    │ Conditional edge — reads guardrail_result.score from state,      │
  │                             │ returns "continue" or "out_of_scope" for LangGraph routing       │
  ├─────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ ainvoke_guardrail_step      │ Async node — LLM scores query, writes guardrail_result to state  │
  └─────────────────────────────┴──────────────────────────────────────────────────────────────────┘
___________________________________________________________________________________________________________
Why temperature=0.0:
    Guardrail decisions must be deterministic. If the same query scores 42 one run
    and 38 the next, the routing changes unpredictably. Zero temperature makes the
    LLM always pick the highest-probability token, ensuring consistent routing.

Why structured output (GuardrailScoring):
    Instead of parsing free text like "Score: 85, reason: ...", with structured_output
    forces the LLM to return valid JSON matching the GuardrailScoring Pydantic model.
    This eliminates parsing errors and makes the score immediately avaialble as
    response.score (int) rather than requiring regex extraction.

Why fallback score=50:
    If the LLM call fails (Ollama timeout, model error), falling back to score=50
    keeps queries in the "borderline" zone - they proceed to retrieval but the system gracefully degrades
    rather than crashing. Better to retrieve and find nothing to fall with a 500 error.
___________________________________________________________________________________________________________

Step-by-step execution trace

state["messages"] = [HumanMessage(content="How does BERT work?")]

Step 1: get_latest_query → "How does BERT work?"
Step 2: start_span("guardrail_validation", input={"query": "How does..."})
Step 3: format GUARDRAIL_PROMPT with question="How does BERT work?"
Step 4: get_langchain_model(temperature=0.0)
Step 5: llm.with_structured_output(GuardrailScoring)
Step 6: structured_llm.ainvoke(prompt)
        → GuardrailScoring(score=88, reason="BERT is a transformer model...")
Step 7: update_span(output={score: 88, decision: "continue"})
Step 8: return {"guardrail_result": GuardrailScoring(score=88, ...)}

LangGraph merges → state["guardrail_result"] = GuardrailScoring(score=88, ...)

Then calls continue_after_guardrail:
score=88 >= threshold=40 → return "continue"
→ graph routes to "retrieve" node

Important: update_span both updates AND ends the span

In PaperAlchemy's LangfuseTracer.update_span(), the implementation calls span.end(...). So calling update_span once on both
success and error paths is correct — it closes the span with the relevant data. No separate end_span() call is needed.
"""

import logging
import time
from typing import Dict, Literal

from langgraph.runtime import Runtime

from src.services.agents.context import Context
from src.services.agents.models import GuardrailScoring
from src.services.agents.prompts import GUARDRAIL_PROMPT
from src.services.agents.state import AgentState
from src.services.agents.nodes.utils import get_latest_query

logger = logging.getLogger(__name__)

def continue_after_guardrail(
        state: AgentState,
        runtime: Runtime[Context],
) -> Literal["continue", "out_of_scope"]:
    """Conditional edge: route based on guardrail score vs threshold.
    What it does:
        Reads guardrail_results from state (set by ainvoke_guardrail_step),
        compares its score against context.guardrail_threshold, and returns
        the routing string LangGraph uses to select the next node.

    Why it is needed:
        LangGraph conditional edges are sync functions that return a string
        key. The graph builder maps that key to a node name:
            graph.add_conditional_edges(
                "guardrail",
                continue_after_guardrail,
                {"continue": "retrieve", "out_of_scope": "out_of_scope"},
            )
        Without this function, there is no way to branch the graph.

    How it helps:
        Keeps routing logic in one place. If the threshold changes
        (e.g., from 40 to 60), only context.guardrail_threshold changes - 
        this function automatically picks up the new value.

    Args:
        state: Current agent state contaning guardrail_result
        runtime: Runtime context with guardrail_thrshold

    Returns:
        "continue" - score >= threshold, proceed to retrieval
        "out_of_scope" - score < threshold, route to rejection node.
    """
    guardrail_result = state.get("guardrail_result")

    if not guardrail_result:
        # Should never happen since ainvoke_guardrail_step always sets a result,
        # but default to continue to avoid silently dropping of valid queries.
        logger.warning("No guardrail_result in state - defaulting to continue")
        return "continue"
    
    score = guardrail_result.score
    threshold = runtime.context.guardrail_threshold

    logger.info(f"Guardrail routing: score={score}, threshold={threshold}")

    return "continue" if score >= threshold else "out_of_scope"

async def ainvoke_guardrail_step(
        state: AgentState,
        runtime: Runtime[Context],
) -> Dict[str, GuardrailScoring]:
    """Async node: score query domain relevance using a structured LLM call.
    What it does:
        1. Extracts the latest HumanMessage from state.messages
        2. Formats GUARDRAIL_PROMPT with the query
        3. Calls ChatOllama with .with_structured_output(GuardrailScoring)
        4. Returns {"guardrail_result": GuardrailScoring(score=..., reason=...)}
        5. Optionally records a LangFuse span of observability

    Why it is needed:
        The first gate in the pipeline. Cheap deterministic filtering before
        any expensive operations (embedding, OpenSearch search, answer generation).

    How it helps:
        - temperature=0.0 -> same query always gets same score (deterministic routing)
        - with structured_output -> score is directly available as int, no parsing
        - Fallback score=50 -> LLM failures don't crash the pipeline
        - Langfuse span -> every guardrail decison is traced for monitoring 

    Args:
        state: Current agent state (reads state["messages])
        runtime: Runtime context (reads ollama_Client, model_name,
                 guardrail_threshold, langfuse_tracer, trace)

    Returns:
        Partial state dict: {"guardrail_result": GuardrailScoring}
        LangGrapgh merges this into the full state automatcially.
    """
    logger.info("NODE: guardrail_validation")
    start_time = time.time()

    # -- Step 1: Extract Query ----------------------------------
    query = get_latest_query(state["messages"])
    logger.debug(f"Evaluation query: {query[:100]}...")

    # -- Step 2: Start LangFuse span (graceful - never raises) -----
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.start_span(
                trace=runtime.context.trace,
                name="guardrail_validation",
                metadata={
                    "node": "guardrail",
                    "model": runtime.context.model_name,
                    "threshold": runtime.context.guardrail_threshold,
                },
                input={"query": query},
            )
        except Exception as e:
            logger.warning(f"Failed to create LangFuse span for guardrail: {e}")

    # --- Step 3: LLM strcutured scoring----------------------------------
    try:
        guardrail_prompt = GUARDRAIL_PROMPT.format(question=query)

        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0, # Deterministic - routing must be consistent
        )
        structured_llm = llm.with_structured_output(GuardrailScoring)

        logger.info("Invoking LLM for guardrail scoring")
        response = await structured_llm.ainvoke(guardrail_prompt)

        logger.info(f"Guardrail result - score: {response.score}, reason: {response.reason}")

        # -- Close span: success path --------------------------------------
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={
                    "score": response.score,
                    "reason": response.reason,
                    "decision": "continue" if response.score >= runtime.context.guardrail_threshold else "out_of_scope",
        
                },
                metadata = {"execution_time_ms": execution_time},
            )
    except Exception as e:
        logger.error(f"Guardrail LLM call failed: {e} - falling back to score=50")

        # Fallback: conservative middle score keeps the query in the pipeline.
        # score 50 is above the default threshold (40), so queries proceed to
        # retrieval: if retrieval finds nothing relevant, grading will route
        # to rewrite - graceful degradation rather than a crash.
        response = GuardrailScoring(
            score=50,
            reason=f"LLM validation failed, suing conservative default: {e}",
        )

        # --- Close span: error path ------------------------------------------
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={
                    "score": response.score,
                    "reason": response.reason,
                    "error": str(e),
                    "fallback": True,
                },
                metadata={"execution_time_ms": execution_time}
            )

    return {"guardrail_result": response}