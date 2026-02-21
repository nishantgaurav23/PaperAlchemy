"""
What is needed:
    The final node in the happy path. It takes the retrieved context from ToolMessage and the user's
    query, formats GENERATE_ANSWER_PROMPT, calls the LLM, and returns the answer as an AIMessage. This
    is the only node that uses temperature=runtime.context.temperature (default 0.7) - all other nodes
    use 0.0

Why it is needed:
    All prior nodes (guardrail, grade, rewrite) produce routing decisions and metadata. This is the only
    node that produces the answer the user actually sees. Without it, the graph completes but produces
    no suseful output.

    It is also the fallback terminal node when retrieve_node hits max_retrieval_attempts - in that case, context
    is empty and the LLM acknowledges no paper were found rather than hallucinating.

How it helps:

Happy path (relevant docs found): 
    context = "BERT uses bidirectional transformers... [paper text]"
    question = "How does BERT work?"
    -> LLM reads papers, synthesizes answer, cites arxiv IDs
    -> AIMessage("BERT (Delvin et al., 2018) uses...") appended to state

Degraded path (max_attempts / no relevant docs):
    context = "" (empty)
    -> node sets context = "No relevant documents found."
    -> LLM honestly says it couldn't find papers
    -> AIMessage("I couldn't find relevant papers...") appended to state

Why start_generation() not start_span()

    PaperAlchemy's LangfuseTracer exposes two different Langfuse primitives:
    - start_span() → trace.span() — for general operations (retrieval, grading)
    - start_generation() → trace.generation() — specifically for LLM calls, tracks token counts and model name in Langfuse's
    generation view

    Since this node calls llm.ainvoke(), using start_generation() gives richer observability — Langfuse shows token usage, model
    name, and cost estimates. The reference uses create_span() for everything, but PaperAlchemy's client has the right tool for
    LLM calls.
________________________________________________________________________________________________________________
What it does: Fnal node in the agentic RAG graph. Takes the retrieved paper context
and user question, formats GENERATE_ANSWER_PROMPT, calls the LLM, and returns the 
answer as an AIMessage. The graph ends after this node.

  ┌──────────────────────────────────┬──────────────────────────────────────────────────────────────┐
  │             Symbol               │                           Purpose                            │
  ├──────────────────────────────────┼──────────────────────────────────────────────────────────────┤
  │ ainvoke_generate_answer_step     │ Async node — LLM synthesises answer from retrieved context   │
  └──────────────────────────────────┴──────────────────────────────────────────────────────────────┘

Why free-form output (not structured):
    Guardrail and grading need structured JSON so routing decisions are type-safe.
    The answer is free-form text — constraining it to a JSON schema would force
    the LLM to put the answer inside a field, adding an unwanted wrapper.
    response.content gives the raw answer string directly.

Why temperature=runtime.context.temperature (not hardcoded):
    Answer generation is the only node where temperature matters for quality.
    Users may want lower temperature for factual, citation-heavy answers or
    higher for more narrative explanations. Making it configurable via
    GraphConfig → Context allows per-request control without changing node code.

Why context fallback to "No relevant documents found.":
    Empty string passed to the LLM prompt makes the {context} placeholder blank,
    confusing small models. An explicit "no documents" string tells the LLM to
    acknowledge the gap rather than hallucinate, producing a more honest response.

Why start_generation() instead of start_span():
    PaperAlchemy's LangfuseTracer.start_generation() maps to trace.generation()
    in the Langfuse SDK — a special span type for LLM calls that tracks model
    name and token usage. The Langfuse UI shows these in a dedicated generations
    view with cost estimates. start_span() would lose this metadata.

__________________________________________________________________________________________________________
Execution trace

state["messages"] = [
    HumanMessage("How does BERT work?"),
    AIMessage(tool_calls=[{retrieve_papers}]),
    ToolMessage('[{"page_content": "BERT uses bidirectional..."}]'),
]
state["relevant_sources"] = [SourceItem(arxiv_id="1810.04805", ...)]

Step 1: question = "How does BERT work?"
Step 2: context = '[{"page_content": "BERT uses bidirectional..."}]'
Step 3: sources_count = 1
Step 4: context is not empty → use as-is
Step 5: start_generation(model="llama3.2:1b", input={question, context_length=...})
Step 6: format GENERATE_ANSWER_PROMPT(context=..., question=...)
Step 7: get_langchain_model(temperature=0.7)  ← context.temperature
Step 8: llm.ainvoke(prompt)
        → AIMessage(content="BERT (Devlin et al., 2018) is a bidirectional...")
Step 9: answer = "BERT (Devlin et al., 2018) is a bidirectional..."
Step 10: update_generation(output=answer, metadata={execution_time_ms=...})
Step 11: return {"messages": [AIMessage(content="BERT (Devlin et al., 2018)...")]}

add_messages appends →
state["messages"][-1] = AIMessage("BERT (Devlin et al., 2018)...")

Graph routes → END
AgenticRAGService._extract_answer() reads last AIMessage → returns to user
"""

import logging
import time
from typing import Dict, List

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.services.agents.context import Context
from src.services.agents.prompts import GENERATE_ANSWER_PROMPT
from src.services.agents.state import AgentState
from src.services.agents.nodes.utils import get_latest_context, get_latest_query

logger = logging.getLogger(__name__)

async def ainvoke_generate_answer_step(
        state: AgentState,
        runtime: Runtime[Context],
) -> Dict[str, List[AIMessage]]:
    """Async node: generate the final answer using retrieved paper context.

    What it does:
        1. Reads the user question and retrieved context from state.messages
        2. Falls back to "No relevant documents found." if context is empty
        3. Formats GENERATE_ANSWER_PROMPT with context + question
        4. Calls ChatOllama (free-form, no structured output)
        5. Extracts response.content as the answer string
        6. Returns {"messages": [AIMessage(content=answer)]}
        7. Records a Langfuse generation span with token tracking

    Why it is needed:
        Every path through the graph (relevant docs found, max attempts reached,
        out_of_scope fallback skipped) eventually needs a terminal answer node.
        This is the canonical answer producer for in-scope queries.

    How it helps:
        - Free-form LLM output → natural, well-structured prose answers
        - GENERATE_ANSWER_PROMPT instructs the LLM to cite arxiv IDs — maps
        directly to SourceItem.arxiv_id in the API response
        - Empty context handled explicitly → honest "not found" responses
        - temperature from context → configurable per-request quality/speed tradeoff
        - Fallback on LLM failure → always returns something to the user

    Args:
        state: Current agent state (reads messages for query and context)
        runtime: Runtime context (reads ollama_client, model_name, temperature,
                relevant_sources count, langfuse config)

    Returns:
        Partial state dict: {"messages": [AIMessage(content=answer)]}
        The add_messages reducer appends this — graph then routes to END.
    """
    logger.info("NODE: generate_answer")
    start_time = time.time()

    # ── Read query and context ──────────────────────────────────────── 
    question = get_latest_query(state["messages"])
    context = get_latest_context(state["messages"])

    sources_count = len(state.get("relevant_sources", []))
    # Explicit empty-context message prevents the LLM seeing a blank {context}
    # placeholder, which causes small models to hallucinate citations.
    if not context:
          logger.warning("No context available for answer generation — using empty-context message")
          context = "No relevant documents found."

    logger.debug(f"Generating answer for: {question[:80]}... | context: {len(context)} chars")

    # ── Start Langfuse generation span ────────────────────────────────
    # Use start_generation (not start_span) — maps to trace.generation()
    # in Langfuse SDK, which tracks token usage and cost in the generations view.
    generation = None
    if runtime.context.langfuse_enabled and runtime.context.trace:
        try:
            generation = runtime.context.langfuse_tracer.start_generation(
                trace=runtime.context.trace,
                name="answer_generation",
                model=runtime.context.model_name,
                input={
                    "question": question,
                    "context_length": len(context),
                    "sources_count": sources_count,
                },
                metadata={
                    "node": "generate_answer",
                    "temperature": runtime.context.temperature,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to create Langfuse generation span: {e}")

    # ── LLM free-form generation ──────────────────────────────────────
    try:
        answer_prompt = GENERATE_ANSWER_PROMPT.format(
            context=context,
            question=question,
        )

        # temperature from context — configurable per request via GraphConfig
        llm = runtime.context.ollama_client.get_langchain_model(
            model=runtime.context.model_name,
            temperature=runtime.context.temperature,
        )

        logger.info("Invoking LLM for answer generation")
        response = await llm.ainvoke(answer_prompt)

        # ChatOllama always returns an AIMessage — .content is the answer string
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"Answer generated: {len(answer)} chars")

        # ── Close generation span: success ────────────────────────────
        if generation:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_generation(
                generation,
                output=answer,
                metadata={
                    "execution_time_ms": execution_time,
                    "answer_length": len(answer),
                    "sources_used": sources_count,
                    "context_length": len(context),
                },
            )

    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}")

        answer = (
            "I apologize — I encountered an error while generating an answer. "
            f"Error: {e}\n\n"
            "Please try again or rephrase your question."
        )

        # ── Close generation span: error ──────────────────────────────
        if generation:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_generation(
                generation,
                output=answer,
                metadata={
                    "execution_time_ms": execution_time,
                    "error": str(e),
                    "fallback": True,
                },
            )

    return {"messages": [AIMessage(content=answer)]}
    

