"""
What is needed:
    A package init file that re-exports every public symbol from the 6 node files and their one conditional
    edge function. This is the single import point that agentic _rag.py (the graph builder) uses.

Why it is needed:                                                                                                            
                                                                                                                              
  Without this file, agentic_rag.py would need 6 individual imports like:

  from src.services.agents.nodes.guardrail_node import ainvoke_guardrail_step, continue_after_guardrail
  from src.services.agents.nodes.retrieve_node import ainvoke_retrieve_step
  from src.services.agents.nodes.grade_documents_node import ainvoke_grade_documents_step
  # ... and so on

  With this file, the graph builder only needs:

  from src.services.agents.nodes import (
      ainvoke_guardrail_step, continue_after_guardrail,
      ainvoke_retrieve_step, ainvoke_grade_documents_step,
      ainvoke_rewrite_query_step, ainvoke_generate_answer_step,
      ainvoke_out_of_scope_step,
  )

  The __all__ list also controls what from nodes import * exposes, preventing accidental import of internal helpers.

How it helps:

  The graph builder in agentic_rag.py calls:
  graph.add_node("guardrail", ainvoke_guardrail_step)
  graph.add_conditional_edges("guardrail", continue_after_guardrail, {...})
  graph.add_node("retrieve", ainvoke_retrieve_step)
  # etc.

  All 7 names come from this single __init__.py. When a node file is renamed or a new node is added, only this file needs
  updating — not the graph builder.

_______________________________________________________________________________________________________________

What it does : 
    Barrel export for all LangGraph agent node functions. Provides a single import
    point so agentic_rag.py can write grapgh without knowing individual file names.

 ┌──────────────────────────────────┬────────────────────┬────────────────────────────────────────┐
  │             Symbol               │       File         │               Role in graph            │
  ├──────────────────────────────────┼────────────────────┼────────────────────────────────────────┤
  │ ainvoke_guardrail_step           │ guardrail_node.py  │ Node function — validates query domain │
  ├──────────────────────────────────┼────────────────────┼────────────────────────────────────────┤
  │ continue_after_guardrail         │ guardrail_node.py  │ Conditional edge — route or reject     │
  ├──────────────────────────────────┼────────────────────┼────────────────────────────────────────┤
  │ ainvoke_out_of_scope_step        │ out_of_scope_node  │ Node function — return rejection msg   │
  ├──────────────────────────────────┼────────────────────┼────────────────────────────────────────┤
  │ ainvoke_retrieve_step            │ retrieve_node.py   │ Node function — trigger tool call      │
  ├──────────────────────────────────┼────────────────────┼────────────────────────────────────────┤
  │ ainvoke_grade_documents_step     │ grade_documents    │ Node function — LLM relevance grading  │
  ├──────────────────────────────────┼────────────────────┼────────────────────────────────────────┤
  │ ainvoke_rewrite_query_step       │ rewrite_query_node │ Node function — rephrase failed query  │
  ├──────────────────────────────────┼────────────────────┼────────────────────────────────────────┤
  │ ainvoke_generate_answer_step     │ generate_answer    │ Node function — produce final answer   │
  └──────────────────────────────────┴────────────────────┴────────────────────────────────────────┘

Why barrel exports:
    agentic_rag.py builds grapgh by calling graph.add_node("name", fn) for each node. It
    only needs the function references - not the internal implementation
    details of each nodel file. This __init__.py decouples the grapgh builder from the node file
    layout.

Why __all__:
    Explicitly controls what is public. Internal helpers in node files (anything
    prefixed with _ or not listed here) are excluded from star imports.

Why the import order is intentional:

    The imports are ordered to reflect the execution flow through the graph:

    guardrail → [out_of_scope | retrieve → grade → [rewrite → retrieve | generate]]

    Reading the imports top-to-bottom gives you the mental model of the graph flow.

    What each exported name is

  ┌──────────────────────────────┬───────────────┬────────────────────────────────────────────────────────┐
  │            Symbol            │     Type      │                       Signature                        │
  ├──────────────────────────────┼───────────────┼────────────────────────────────────────────────────────┤
  │ ainvoke_guardrail_step       │ async node fn │ (state, runtime) → dict                                │
  ├──────────────────────────────┼───────────────┼────────────────────────────────────────────────────────┤
  │ continue_after_guardrail     │ sync edge fn  │ (state, runtime) → Literal["continue", "out_of_scope"] │
  ├──────────────────────────────┼───────────────┼────────────────────────────────────────────────────────┤
  │ ainvoke_out_of_scope_step    │ async node fn │ (state, runtime) → dict                                │
  ├──────────────────────────────┼───────────────┼────────────────────────────────────────────────────────┤
  │ ainvoke_retrieve_step        │ async node fn │ (state, runtime) → dict                                │
  ├──────────────────────────────┼───────────────┼────────────────────────────────────────────────────────┤
  │ ainvoke_grade_documents_step │ async node fn │ (state, runtime) → dict                                │
  ├──────────────────────────────┼───────────────┼────────────────────────────────────────────────────────┤
  │ ainvoke_rewrite_query_step   │ async node fn │ (state, runtime) → dict                                │
  ├──────────────────────────────┼───────────────┼────────────────────────────────────────────────────────┤
  │ ainvoke_generate_answer_step │ async node fn │ (state, runtime) → dict                                │
  └──────────────────────────────┴───────────────┴────────────────────────────────────────────────────────┘

Note: continue_after_guardrail is synchronous — LangGraph calls conditional edge functions synchronously to decide routing.
All node functions are async because they call await llm.ainvoke(...) or await embeddings_client.embed_query(...).
"""

# Nodes written so far (Files 8-9)
from .guardrail_node import ainvoke_guardrail_step, continue_after_guardrail
from .out_of_scope_node import ainvoke_out_of_scope_step

# Uncomment each line as the corresponding file is written (Files 10-13):
# from .retrieve_node import ainvoke_retrieve_step
# from .grade_documents_node import ainvoke_grade_documents_step
# from .rewrite_query_node import ainvoke_rewrite_query_step
# from .generate_answer_node import ainvoke_generate_answer_step

__all__ = [
    "ainvoke_guardrail_step",
    "continue_after_guardrail",
    "ainvoke_out_of_scope_step",
    # Uncomment as files are added:
    # "ainvoke_retrieve_step",
    # "ainvoke_grade_documents_step",
    # "ainvoke_rewrite_query_step",
    # "ainvoke_generate_answer_step",
]