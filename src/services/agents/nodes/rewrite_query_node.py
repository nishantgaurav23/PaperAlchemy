"""
What is needed

A node that uses an LLM to produce a better search query when grading determined the retrieved documents were not relevant.
It adds a new HumanMessage with the rewritten query to state["messages"] and sets rewritten_query in state. The graph then
loops back to retrieve_node.

Why it is needed

When grade_documents_node routes to "rewrite_query", the original query failed to find relevant papers. Common reasons:
- Query too vague ("attention" → should be "self-attention transformer architecture")
- Wrong terminology ("memory model" → should be "LSTM long short-term memory network")
- Too broad ("deep learning" → should be "convolutional neural network image classification")

The LLM understands semantic intent and can expand jargon, add technical synonyms, and rephrase for retrieval — things a
keyword system can't do.

How it helps

Round 1: query = "how does GPT work?"
    → retrieve → grade → "not relevant" (retrieval too vague)
    → rewrite_query: "GPT generative pre-trained transformer autoregressive language model"
    → new HumanMessage appended to state.messages
    → retrieve_node: get_latest_query() returns the REWRITTEN query
    → retrieve (round 2) → grade → "relevant" → generate_answer

The key is that add_messages appends the HumanMessage(rewritten_query) to state. get_latest_query() scans messages in
reverse, so the next nodes automatically pick up the rewrite without any special routing.

Why QueryRewriteOutput is defined locally (not in models.py)

models.py contains shared models used across multiple nodes. QueryRewriteOutput is only ever used inside
rewrite_query_node.py — no other node needs it. Keeping it local avoids polluting the shared models namespace with a
single-use type.

_____________________________________________________________________________________________________
What it does: Uses a structured LLM call to rewrite the user's query into a version
  that will retrieve more relevant documents. Adds the rewritten query as a new
  HumanMessage so get_latest_query() returns it on the next retrieve attempt.

  ┌───────────────────────────────┬────────────────────────────────────────────────────────────────┐
  │            Symbol             │                           Purpose                              │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ QueryRewriteOutput            │ Local Pydantic model for structured LLM rewrite output         │
  ├───────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ ainvoke_rewrite_query_step    │ Async node — LLM rewrites query, adds HumanMessage to state    │
  └───────────────────────────────┴────────────────────────────────────────────────────────────────┘

Why return a new HumanMessage (not just set rewritten_query):
    The add_messages reducer on state["messages"] APPENDS the new HumanMessage.
    This means get_latest_query() — which scans messages in reverse — will return
    the rewritten query on the next call. Nodes that need the query (retrieve,
    grade, generate) work correctly without needing to know about rewriting.
    Setting rewritten_query separately is for tracing/API response only.

Why use original_query (not the latest HumanMessage):
    On the second rewrite (if the first rewrite also fails), state["messages"]
    contains multiple HumanMessages. Rewriting the rewritten query would cause
    semantic drift — each pass moves further from the user's intent.
    Using original_query ensures every rewrite starts from the user's actual words.

Why temperature=0.3 (not 0.0):
    Query rewriting benefits from slight creativity — synonyms, domain expansions,
    alternative phrasings. Pure greedy decoding (0.0) tends to produce minimal
    changes. 0.3 gives enough variance to actually improve retrieval while staying
    focused on the original intent.

Why QueryRewriteOutput is local (not in models.py):
    Only this node uses it. Adding it to the shared models namespace would create
    a false impression that other nodes depend on it.
"""

import logging
import time
from typing import Dict, List, Union

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from src.services.agents.context import Context
from src.services.agents.prompts import REWRITE_PROMPT
from src.services.agents.state import AgentState

logger = logging.getLogger(__name__)

class QueryRewriteOutput(BaseModel):
    """Structured output for LLM-based query rewriting.
    What it does:
        Defines the JSON schema the LLM must return when called with
        with_structured output(QueryRewriteOutput). Forces the model to
        produce both the improved query and reasoning for tracing.

    Why local (not it models.py):
        Only rewrite_query_node uses this. Single-use types belong in
        the file that uses them.
    """

    rewritten_query: str = Field(
        description="The improved query optimized for documents retrieval"
    )
    reasoning: str = Field(
        description="Brief explanation of how the query was improved"
    )

async def ainvoke_rewrite_query_step(
            state: AgentState,
            runtime: Runtime[Context],
    ) -> Dict[str, Union[str, List[HumanMessage]]]:
        """Async node: rewrite the user query using LLM for better retrieval.
        
        What it does:
            1. Reads original_query from state (falls back to first message)
            2. Formats REWRITE_PROMPT with the orignal question
            3. Calls ChatOllama with .with_strcutured_output(QueryRewriteOutput)
            4. Validates the result is non-empty
            5. Falls back to keyword expansion if LLM fails

        Why it is needed:
            After grade_documents routes to "rewrite_query", the current query
            failed retrieval. The LLM understands semantic intent and can expand
            technical terminology , add syninyms, and restrcuture the query for better
            vector + BM25 retrieval.

        How it helps:
            - Always rewrites from original_query — prevents semantic drift on
              multiple rewrite rounds
            - with_structured_output gives structured access to both the query
              (for retrieval) and reasoning (for tracing)
            - Fallback appends "research paper arxiv machine learning" to ensure
              retrieval can still run even if the LLM fails
            - New HumanMessage integrates transparently with get_latest_query()

        Args:
            state: Current agent state (reads original_query, retrieval_attempts)
            runtime: Runtime ocntext (reads ollama_client, model_name, langfuse config)

        Returns:
            Partial state dict:
            - messages: [HumanMessage(rewritten_query)] - appended via add_messages
            - rewritten_query: str - stored separately for API response / tracing
        """
        logger.info(f"NODE: rewrite_query")
        start_time = time.time()

        # ── Read original query ───────────────────────────────────────────
        # Always rewrite from the original — not from a previous rewrite.
        # Fallback to first message content if original_query was never set
        # (defensive: retrieve_node should always set it on attempt 1).
        original_question = (
            state.get("original_query")
            or state["messages"][0].content
        )
        current_attempt = state.get("retrieval_attempts", 0)

        logger.debug(f"Rewriting (attempt {current_attempt}): {original_question[:100]}...")

        # --- Start Langfuse span -----------------------------------------------------------
        span = None
        if runtime.context.langfuse_enabled and runtime.context.trace:
            try:
                span = runtime.context.langfuse_tracer.start_span(
                    trace=runtime.context.trace,
                    name="query_rewritting",
                    metadata={
                        "node": "rewrite_query",
                        "model": runtime.context.model_name,
                        "strategy": "llm_structured_rewrite",
                        "attempt": current_attempt,
                    },
                    input={"orignal_query": original_question},

                )
            except Exception as e:
                logger.warning(f"Failed to create LangFuse span for rewrite_query: {e}")

        # ------- LLM structured rewriting----------------------------------------------------
        llm_duration: float = 0.0
        try:
            llm = runtime.context.ollama_client.get_langchain_model(
                model=runtime.context.model_name,
                temperature=0.3, #Slightly creativity for synonym/expansion variety
            )
            structured_llm = llm.with_structured_output(QueryRewriteOutput)

            prompt = REWRITE_PROMPT.format(question=original_question)

            logger.info(f"Invoking LLM for query rewriting (model: {runtime.context.model_name})")
            llm_start = time.time()
            result: QueryRewriteOutput = await structured_llm.ainvoke(prompt)
            llm_duration = time.time() - llm_start

            # Validate - with structured_output can return a valid model with empty strings
            if not result or not result.rewritten_query or not result.rewritten_query.strip():
                raise ValueError("LLM returned empty rewritten_query")
            
            rewritten_query = result.rewritten_query.strip()
            reasoning = result.reasoning

            logger.info(
                f"Rewrite complete in {llm_duration:.2f}s: "
                f"'{original_question[:50]}' -> '{rewritten_query[:50]}"
            )
        except Exception as e:
            logger.error(f"LLM query rewriting failed: {e} - falling back to keyword expansion")

            # Fallback: append standard retrieval keywords to improve BM25 matching.
            # Crude but ensures the next retrieve attempt uses different terms.
            rewritten_query = f"{original_question} research paper arxiv machine learning"
            reasoning = f"Fallback: keyword expansion (LLM failed): {e}"

        # --- Close Langfuse span ----------------------------------------------------------
        if span:
            execution_time = (time.time() - start_time) * 1000
            runtime.context.langfuse_tracer.update_span(
                span,
                output={
                    "rewritten_query": rewritten_query,
                    "reasoning": reasoning,
                    "original_query": original_question,
                },
                metadata={
                    "execution_time_ms": execution_time,
                    "original_length": len(original_question),
                    "rewritten_length": len(rewritten_query),
                    "llm_duration_seconds": llm_duration, # 0.0 in fallback path - no NameError
                },
            )
        # ── Return new HumanMessage ───────────────────────────────────────
        # add_messages reducer APPENDS this — does not replace the original.
        # get_latest_query() will now return rewritten_query on next nodes.
        return {
            "messages": [HumanMessage(content=rewritten_query)],
            "rewritten_query": rewritten_query,
        }
