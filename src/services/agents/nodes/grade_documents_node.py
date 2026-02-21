"""
What is needed:
    A node that uses a structured LLM to decide whether the retrieved documents are relevant to the user's query.
    It sets routing_decision in state to either "generate_answer" (relevant) or "rewrite_query" (not relevant).
    and records a GradingResult for the reasoning trace.

Why it is needed:
    After ToolNode executes retrieve_papers, the documents land in state["messages"] as a
    ToolMessage. But having documents doesn't mean they're useful - OpenSearch may return
    papers that are semantically adjacent but don't actually answer the question.
    The grade node filters this noises:
    - Relevant -> proceed to generate_answer with high-quality context
    - Not relevant -> route to reqrite_query for better search attempt

    Without this node, the grapgh would always generate an answer with whatever was
    retrieved - even retrieval failed completely.

How it helps:
Query: "What is attention mechanism in transformers?"
Retrieved: ["BERT paper", "GPT-2 paper"] → binary_score = "yes"
    → routing_decision = "generate_answer"
    → generate_answer uses these papers as context

Query: "attention mechanism"  (vague, poor retrieval)
Retrieved: ["Paper on human attention psychology"] → binary_score = "no"
    → routing_decision = "rewrite_query"
    → rewrite_query produces "self-attention mechanism transformer neural network"
    → retrieve_node tries again with better query 

_________________________________________________________________________________
What it does: Uses a structured LLM call (binary yes/no) to decide whether retrieved
  documents are relevant to the query. Sets routing_decision to "generate_answer" or
  "rewrite_query" and records a GradingResult for the reasoning trace.

  ┌──────────────────────────────────┬────────────────────────────────────────────────────────
  ──────┐
  │             Symbol               │                           Purpose
        │
  ├──────────────────────────────────┼────────────────────────────────────────────────────────
  ──────┤
  │ ainvoke_grade_documents_step     │ Async node — LLM grades retrieved docs, sets
  routing_decision│
  └──────────────────────────────────┴────────────────────────────────────────────────────────
  ──────┘

Why binary grading (not per-document):
    The reference and PaperAlchemy both grade the entire retrieved context as a
    unit rather than scoring each document individually. This is faster (one LLM
    call instead of top_k calls) and matches how the generate node consumes context
    — as a single concatenated string, not individual chunks.

Why temperature=0.0:
    Grading is a routing decision. "yes" today must be "yes" tomorrow for the
    same inputs. Deterministic scoring prevents the graph from taking different
    paths on identical queries, which would make debugging and tracing impossible.

Why fallback to heuristic (len > 50):
    If the LLM call fails, we need a routing decision to avoid deadlock. The
    heuristic "more than 50 chars of context = probably relevant" is imperfect
    but better than crashing or routing to rewrite unconditionally (which wastes
    an attempt slot).

Conditional edge (defined in agentic_rag.py, not here):
    After this node, the graph reads state["routing_decision"] to pick the next
    node. The conditional edge function in agentic_rag.py maps:
        "generate_answer" → generate_answer node
        "rewrite_query"   → rewrite_query node
___________________________________________________________________________________
Execution trace

state["messages"] = [
    HumanMessage("How does BERT work?"),
    AIMessage(tool_calls=[{retrieve_papers}]),
    ToolMessage(content='[{"page_content": "BERT is a...", "metadata": {...}}]'),
]

Step 1: question = "How does BERT work?"
Step 2: context = '[{"page_content": "BERT is a..."}]'  ← from ToolMessage
Step 3: context is not empty → skip fast path
Step 4: format GRADE_DOCUMENTS_PROMPT(context=..., question=...)
Step 5: get_langchain_model(temperature=0.0)
Step 6: llm.with_structured_output(GradeDocuments)
Step 7: structured_llm.ainvoke(prompt)
        → GradeDocuments(binary_score="yes", reasoning="BERT is directly relevant...")
Step 8: is_relevant = True, score = 1.0
Step 9: route = "generate_answer"
Step 10: return {
    "routing_decision": "generate_answer",
    "grading_results": [GradingResult(is_relevant=True, score=1.0, ...)]
}

Graph reads routing_decision → routes to generate_answer node
"""

import logging
import time
from typing import Dict, List, Union

from langgraph.runtime import Runtime

from src.services.agents.context import Context
from src.services.agents.models import GradeDocuments, GradingResult
from src.services.agents.prompts import GRADE_DOCUMENTS_PROMPT
from src.services.agents.state import AgentState
from src.services.agents.nodes.utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)

async def ainvoke_grade_documents_step(
        state: AgentState,
        runtime: Runtime[Context],
) -> Dict[str, Union[str, List[GradingResult]]]:
    """Async node: grade retrived docments and set routing_decisions.
    
    What it does:
        1. Reads the urrent query and retrieved context from state.messages
        2. If no context: immediately routes to rewrite_query (no LLM needed)
        3. Format GRDADE_DOCUMENTS_PROMPT with context + query
        4. Calls ChatOllama with .with_structured_output(GradedDocuments)
        5. Maps binary_Score -> routing_decisions + GradingResult
        6. Falls back to a length heuristic if the LLM call fails.
        7. Optionally records a LangFuse span.

    Why it is needed:
        Retrieval quality is imperfect. Without grading, every retrieval result
        goes directly to answer generation - even empty or off-topic results.
        This node creates the feedback loop (retrieve -> grade -> rewrite) that 
        progressively improves retrieval quality.

    How it helps:
        - temperature=0.0 -> deterministic routing (same query -> same route)
        - with_structured_output(GradeDocuments) → binary_score is a typed
          Literal["yes", "no"], not a string to parse
        - No-context fats path -> avoids LLM call when retrieval clearly failed
        - Heuristic fallback -> LLM failure doesn't deadlock the graph

    Args:
        state: Current agent state (reads message for query and context)
        runtime: Runtime context (reads ollama_client, model_name, langfuse config)

    Returns: 
        Partial state dict:
        - routing_decision: "generate_answer" or "rewrite_query"
        - grading_results: [GradingResult] with score, reasoning, is_relevant
    """
    logger.info("Node: grade_documents")
    start_time = time.time()

    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    # ---------- Start Langfuse span ----------------------------------------------
    span = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            span = runtime.context.langfuse_tracer.start_span(
                trace=runtime.context.trace,
                name="document_grading",
                metadata={
                    "node": "grade_documents",
                    "model": runtime.context.model_name,
                    "context_length": len(context) if context else 0,
                    "has_context": bool(context),
                },
                input={"query": question},

            )
        except Exception as e:
            logger.warning(f"Failed to create LangFuse span for grade_documents: {e}")

    # ---- Fast path: no context retrieved -----------------------------------
    # ToolNode ran but retrieved nothing - no LLM call needed, just rewrite.
    if not context:
        logger.warning("No context found in messages - routing to rewrite_query")

        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={"routing_decision": "rewrite_query", "grading_results": []},
                metadata={"execution_time_ms": execution_time},
            )
        
        return {"routing_decision": "rewrite_query", "grading_results": []}
    
    # ------LLM structured grading ---------------------------------------------
    logger.debug(f"Grading {len(context)} chars of context for: {question[:80]}...")

    try:
        grading_prompt = GRADE_DOCUMENTS_PROMPT.format(
            context=context,
            question=question,
        )

        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=0.0, # Deterministic - routing must be consistent
        )
        structured_llm = llm.with_strcutured_output(GradeDocuments)

        logger.info("Invoking LLM for document grading")
        grading_response = await structured_llm.ainvoke(grading_prompt)

        is_relevant = grading_response.bianry_score == "yes"
        logger.info(
            f"Grading result: binary_score={grading_response.binary_score}, "
            f"reasoning={grading_response.reasoning[:100]}..."
        )

        grading_result = GradingResult(
            document_id="retrieved_docs",
            is_relevant=is_relevant,
            score=1.0 if is_relevant else 0.0,
            reasoning=grading_response.reasoning,
        )

    except Exception as e:
        logger.error(F"LLM grading failed: {e} - falling back to length heuristic")

        # Heuristic: if we got more than 50 chars, assume it has some content.
        # Imperfect, but prevents the graph from deadlocking on LLM failure.
        is_relevant = len(context.strip()) > 50
        grading_result = GradingResult(
            document_id="retrieved_docs",
            is_relevant=is_relevant,
            score=1.0 if is_relevant else 0.0,
            reasoning=(
                f"Fallback heurisitc (LLM failed: {e}): "
                f"{'sufficient contect' if is_relevant else 'insufficient content'}"
            ),
        )

    # ---- Set routing decision------------------------------------------------
    route = "generate_answer" if is_relevant else "rewrite_query"
    logger.info(f"Routing to: {route}")

    # --------Close LangFuse span ---------------------------------------------
    # Use grading_result.score (always defined) not a local `score` variable
    # to avoid NameError in the exception path.
    if span:
        execution_time = (time.time() - start_time) * 1000
        runtime.context.langfuse_tracer.update_span(
            span,
            output={
                "routing_decision": route,
                "is_relevant": is_relevant,
                "score": grading_result.score,
                "reasoning": grading_result.reasoning,
            },
            metadata={
                "execution_time_ms": execution_time,
                "context_length": len(context)
            },
        )

    return {
        "routing_decision": route,
        "grading_results": [grading_result],
    }

