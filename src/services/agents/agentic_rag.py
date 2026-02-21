"""
What is needed:
    The top-level orchestrator that wires together all 6 node files, the retriever
    tool, and LangGraph's StateGrapgh into a runnable workflow. This is the only file
    that tocuhes the graph API directly - every other file is a pure component.

Why it is needed:
    LangGraph graphs must be explicitly constructed : nodes declared, edges wired, 
    conditional routing functions registered, and the workflow compiled before it
    can accept input. Withput this file, all the individual nodes are just functions
    with no connection between them.

    The file also owns the Langfuse trace lifecycle. Each call to ask() creates one
    trace; every node creates child spans under it. The trace is closed here after
    thhe grapgh finishes.

How it helps:
    - AgenticRAGService is constructed ONCE at startup (dependency injection)
    - The compiled graph is reused across all requests (no rebuld per request)
    - ask() creates a per-request Context carrying runtime config + trace handle
    - _run_workflow() does the actual graph invocation and result extraction
    - _extract_* helpers parse raw state into clean API-ready types

Request flow:
    ask(query) -> create_trace -> _run_workflow -> graph.ainvoke
    -> _extract_answer / _extarct_sources / _extarct_reasoning_steps
    -> return dict

__________________________________________________________________________________________________________________
What it does:
    Defines AgenticRAGService - the class the FastAPI router calls to handle
    agentic RAG requests. Builds a compiled LangGraph workflow in __init__ and invokes
    it per request in ask()

 ┌──────────────────────────────────┬────────────────────────────────────────────────────────────┐
   │            Method                │                        Purpose                             │
   ├──────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ __init__                         │ Injects service clients, builds + compiles graph once       │
   ├──────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ _build_graph                     │ Declares all nodes, edges, and routing logic                │
   ├──────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ ask                              │ Entry point: validates query, manages trace, calls workflow │
   ├──────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ _run_workflow                    │ Creates Context, invokes graph, returns structured result   │
   ├──────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ _extract_answer                  │ Reads last AIMessage from state["messages"]                 │
   ├──────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ _extract_sources                 │ Parses ToolMessages → list[dict] via utils helper           │
   ├──────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ _extract_reasoning_steps         │ Builds human-readable steps from state fields               │
   ├──────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ get_graph_visualization          │ Returns PNG bytes of the compiled graph                     │
   ├──────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ get_graph_mermaid                │ Returns mermaid syntax string of the compiled graph         │
   ├──────────────────────────────────┼────────────────────────────────────────────────────────────┤
   │ get_graph_ascii                  │ Returns ASCII art representation of the graph               │
   └──────────────────────────────────┴────────────────────────────────────────────────────────────┘

 ____________________________________________________________________________________________________________
 Why _build_graph is called once in __init__ (not per request):
    Compiling a LangGraph workflow involves Python object construction and edge
    validation. Doing it per request would add ~50ms overhead per call for zero
    benefit - the graph structure never changes between requests. only the Context
    (injected at runtime) varies per request.

Why context_schema=Context:
    LangGraph's context_schema parameter enables dependency into nodes.
    Instead of closures or global state, every node receives a typed 
    Context instance at runtime via the `runtime` parameter: `runtime`:
    Runtime[Context]`. This makes nodes pure, testable functions.

Why nodes use Runtime[Context] not Context directly:
    LangGraph wraps the context in a Runtime object for forward compatibility. 
    Nodes access dependencies via `runtime.context.ollama_client` etc.

Langfuse trace pattern (PaperAlchemy vs reference):
    PaperAlchemy:        langfuse_tracer.create_trace(name=..., metadata=...)

    PaperAlchemy's LangfuseTracer.create_trace() calls self._langfuse.trace(...)
    which returns the raw Langfuse trace object. This object has .update() and
    .end() methods directly (from the Langfuse Python SDK). We call those directly
    since LangfuseTracer doesn't wrap them.

    Nodes do their own child spans via runtime.context.langfuse_tracer.start_span()
    and runtime.context.langfuse_tracer.update_span(). No CallbackHandler needed.

Why _extract_sources uses extarct_sources_from_tool_messages (not state["relevant_sources"]):
    No node explicitly writes to state["relevant_sources"]. The actual retrieved
    papers live in ToolMessages added to state["messages] by LangGraph's ToolNode.
    extract_sources_from_tool_messages parses those ToolMessages into SourceItem object

____________________________________________________________________________________________________________
Graph structure:

     START
       │
       ▼
     guardrail ──(score < threshold)──► out_of_scope ──► END
       │
       │ (score >= threshold)
       ▼
     retrieve ──(tool_calls)──► tool_retrieve ──► grade_documents
       ▲                                              │
       │                                              │ (irrelevant)
       │                                              ▼
       └──────────────────── rewrite_query ◄──────────┘
                                                       │ (relevant)
                                                       ▼
                                               generate_answer ──► END
 ____________________________________________________________________________________________________________

Function-by-function summary

Function: _build_graph
Called by: __init__
Returns: Compiled LangGraph graph
Key design decision: context_schema=Context enables typed dependency injection

Function: ask
Called by: FastAPI router / tests
Returns: dict with answer, sources, reasoning_steps, metadata
Key design decision: Validates query, creates trace, delegates to _run_workflow

Function: _run_workflow
Called by: ask
Returns: dict (same shape as ask return)
Key design decision: State initialisation + Context construction + graph invocation

Function: _extract_answer
Called by: _run_workflow
Returns: str
Key design decision: Reads messages[-1].content — works for both generate and out_of_scope nodes

Function: _extract_sources
Called by: _run_workflow
Returns: list[dict]
Key design decision: Uses extract_sources_from_tool_messages (not state["relevant_sources"])

Function: _extract_reasoning_steps
Called by: _run_workflow
Returns: list[str]
Key design decision: Builds strings from guardrail_result, retrieval_attempts, grading_results
"""

import logging
import time
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.opensearch.client import OpenSearchClient

from .config import GraphConfig
from .context import Context
from .nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
    continue_after_guardrail,
)
from .nodes.utils import extract_sources_from_tool_messages
from .state import AgentState
from .tools import create_retriever_tool

logger = logging.getLogger(__name__)

class AgenticRAGService:
    """Orchestrates the agentic RAG LangGraph workflow.
    
    What it does:
        Builds a compiled LangGraph workflow at init time and exposes ask()
        as the single entry point for the FastAPI router. Each ask() call
        runs the full pipeline: guardrail → retrieve → grade → generate.

    Why it is needed:
        LangGraph requires explicit graph construction (nodes, edges, routing)
        before execution. This class encapsulates that construction so the
        router only needs to call ask(query) without knowing graph internals.

    How it helps:
        - Graph compiled once at startup, reused across all requests
        - Context injected per request — different model/config per call
        - Langfuse trace created and closed per request for observability
        - _extract_* methods convert raw LangGraph state to clean API types
    """

    def __init__(
            self,
            opensearch_client: OpenSearchClient,
            ollama_client: OllamaClient,
            embeddings_client: JinaEmbeddingsClient,
            langfuse_tracer: Optional[LangfuseTracer] = None,
            graph_config: Optional[GraphConfig] = None,
    ):
        """Initialise the service and compile the LangGraph workflow.
        
        What it does:
            Stores injected service clients, resolves config defaults,
            and calls _build_graph() to compile the workflow once.

        While compile in __init__:
            Graph structure is constant across requests. Compiling once
            here vs per-request saves ~50ms overhead and avoids recreating
            the ToolNode (which wraps the retriever tool) on every call.

        Args:
            opensearch_client: Document search backend
            ollama_client: LLM backend for guardrail, grading, answer generation
            embeddings_client: Jina client for query embedding (used by retriever tool)
            langfuse_tracer: Optional tracing wrapper - None disables all tracing
            graph_config: Workflow config (model, top_k, thresholds). Defaults applied
                          from settings if None.
        """
        self.opensearch =  opensearch_client
        self.ollama = ollama_client
        self.embeddings = embeddings_client
        self.langfuse_tracer = langfuse_tracer
        self.graph_config = graph_config or GraphConfig()

        logger.info("Initialising AgenticRAGService:")
        logger.info(f" Model: {self.graph_config.model}")
        logger.info(f"  Top-k: {self.graph_config.top_k}")
        logger.info(f"  Hybrid search: {self.graph_config.use_hybrid}")
        logger.info(f"  Max retrieval attempts: {self.graph_config.max_retrieval_attempts}")
        logger.info(f"  Guardrail threshold: {self.graph_config.guardrail_threshold}")

        self.graph = self._build_graph()
        logger.info("AgenticRAGService initialised successfully")

    def _build_graph(self):
        """Build and complete the LangGraph workflow.
        
        What it does:
            1. Creates a StateGraph with AgentState and Context schema
            2. Creates the retriever tool (captures service clients via closure)
            3. Registers all 6 nodes + ToolNode
            4. Wires edges and conditional routing
            5. Compiles and returns the runnable graph

        Why context_schema=Context:
            Enables LangGraph's typed dependency injection. Every node declared
            inside this graph will recieve a `runtime: Runtime[Context]` argument
            populated with the Context instance passed to graph.ainvoke().

        Why ToolNode vs a custome node:
            LangGraph's ToolNode is the standard way to execute @tool-decorated
            functions. It reads the tool_calls from the last AIMessage, invokes
            the matching @tool, and appends a ToolMessage to state["messages"].
            No custom serialixation or routing needed.

        Returns:
            Compiled graph ready for graph.ainvoke(state, config, context=ctx)
        """
        logger.info("Building LangGraph workflow")

        workflow = StateGraph(AgentState, context_schema=Context)

        # Retriever tool: closure captures opensearch + embeddings clients.
        # ToolNode registers it so LangGraph can dispatch it by name.
        retriever_tool = create_retriever_tool(
            opensearch_client=self.opensearch,
            embeddings_client=self.embeddings,
            top_k=self.graph_config.top_k,
            use_hybrid=self.graph_config.use_hybrid,
        )
        tools = [retriever_tool]

        # ----------Nodes--------------------------------------------------------------
        logger.info("Registered nodes")
        workflow.add_node("guardrail", ainvoke_guardrail_step)
        workflow.add_node("out_of_scope", ainvoke_out_of_scope_step)
        workflow.add_node("retrieve", ainvoke_retrieve_step)
        workflow.add_node("tool_retrieve", ToolNode(tools))
        workflow.add_node("grade_documents", ainvoke_grade_documents_step)
        workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
        workflow.add_node("generate_answer", ainvoke_generate_answer_step)

        # ── Edges ─────────────────────────────────────────────────────────
        logger.info("Wiring edges and routing logic")

        # Entry point
        workflow.add_edge(START, "guardrail")

        # Guardrail -> route by domain relevance score
        workflow.add_conditional_edges(
            "guardrail",
            continue_after_guardrail,
            {
                "continue": "retrieve",
                "out_of_scope": "out_of_scope"
            },
        )

        # Out-of-scope is terminal
        workflow.add_edge("out_of_scope", END)

        # Retrieve → tools_condition checks if last AIMessage has tool_calls.
        # "tools" → ToolNode executes retrieve_papers.
        # END → retrieve_node hit max attempts and returned fallback AIMessage.

        workflow.add_conditional_edges(
            "retrieve",
            tools_condition,
            {
                "tools": "tool_retrieve",
                END: END,
            },
        )

        # After retrieval -> grade document relevance
        workflow.add_edge("tool_retrieve", "grade_documents")

        # Grade -> route based on relevance decision set in state["routing_decision"]
        workflow.add_conditional_edges(
            "grade_documents",
            lambda state: state.get("routing_decision", "generate_answer"),
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
            },
        )

        # Rewrite -> retry retrieve with imporved query
        workflow.add_edge("rewrite_query", "retrieve")

        # Generate -> done
        workflow.add_edge("generate_answer", END)

        logger.info("Compiling workflow")
        compiled_graph = workflow.compile()
        logger.info("Graph compilation successful")
        return compiled_graph

    async def ask(
            self,
            query: str,
            user_id: str = "api_user",
            model: Optional[str] = None,
    ) -> dict:
        """Ask a question using agentic RAG.

        What it does:
            1. Validates the query is non-empty
            2. Creates a Langfuse trace for the full request (if enabled)
            3. Delegates to _run_workflow for graph execution
            4. Returns structured result: answer, sources, reasoning, metadata

        Why Langfuse trace at this level:
            The trace represents the entire agentic request. Child spans
            (guardrail, grade, generate) are created inside each node under
            this trace. Closing it here means the trace is complete when ask()
            returns — correct for the Langfuse timeline view.

        Args:
            query: User question string (must be non-empty)
            user_id: Identifier for Langfuse user_id field (default "api_user")
            model: Optional model override — uses graph_config.model if None

        Returns:
            dict with keys:
                query:              Original user query
                answer:             Generated answer string
                sources:            List of source dicts (arxiv_id, title, url, ...)
                reasoning_steps:    List of human-readable step strings
                retrieval_attempts: How many retrieve→grade loops ran
                rewritten_query:    Rewritten query if rewrite happened, else None
                execution_time:     Wall-clock seconds for the full pipeline
                guardrail_score:    Integer 0-100 domain relevance score

        Raises:
            ValueError: If query is empty
        
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        model_to_use = model or self.graph_config.model

        logger.info("=" * 70)
        logger.info("Agentic RAG request starting")
        logger.info(f" Query: {query[:100]}")
        logger.info(f" User: {user_id} | Model: {model_to_use}")
        logger.info("=" * 70)

        # -- Langfuse trace (one per request) --------------------------------
        trace = None
        if self.langfuse_tracer and self.langfuse_tracer.enabled:
            try:
                trace = self.langfuse_tracer.create_trace(
                    name="agentic_rag_request",
                    metadata={
                        "service": "agentic_rag",
                        "model": model_to_use,
                        "top_k": self.graph_config.top_k,
                        "use_hybrid": self.graph_config.use_hybrid,
                    },
                    user_id=user_id,
                    session_id=f"session_{user_id}",
                    input={"query": query},
                )
                logger.debug("LangFuse trace created")
            except Exception as e:
                logger.warning(f"Failed to create Langfuse trace: {e}")

        try:
            return await self._run_workflow(query, model_to_use, user_id, trace)
        except Exception as e:
            logger.error(f"Agentic RAG request failed: {e}")
            if trace:
                try:
                    trace.update(output={"error": str(e)}, level="ERROR")
                except Exception:
                    pass
                self.langfuse_tracer.flush()
            raise

    async def _run_workflow(
         self,
         query: str,
         model_to_use: str,
         user_id: str,
         trace,
     ) -> dict:
        """Execute the compiled LangGraph workflow and return structured results.

        What it does:
            1. Builds the initial AgentState with all fields at their defaults
            2. Constructs the per-request Context with service clients + config
            3. Calls graph.ainvoke(state, config, context=runtime_context)
            4. Extracts answer, sources, reasoning steps from the final state
            5. Updates and closes the Langfuse trace with results

        Why state initialisation here (not in the graph):
            LangGraph requires state to be provided at invocation time. Setting
            all fields to known defaults here prevents KeyError in nodes that
            do state.get("retrieval_attempts", 0) — the default is explicit.

        Why Context is created per request:
            model_name and trace can vary per request. Context is a lightweight
            dataclass (slots=True) — instantiation cost is negligible.

        Args:
            query: Validated query string
            model_to_use: Resolved model name (from request or graph_config)
            user_id: User identifier (for logging only at this level)
            trace: Active Langfuse trace object, or None if disabled

        Returns:
            Structured result dict — same shape as ask() docstring describes
        """
        start_time = time.time()

        # -- Initial state: all fields at defaults --------------------------
        state_input = {
            "messages": [HumanMessage(content=query)],
            "retrieval_attempts": 0,
            "guardrail_result": None,
            "routing_decision": None,
            "sources": None,
            "relevant_sources": [],
            "relevant_tool_artefacts": None,
            "grading_results": [],
            "metadata": {},
            "original_query": None,
            "rewritten_query": None,
        }

        # -- Runtime context: injected into every node via Runtime[Context] -
        runtime_context = Context(
            ollama_client=self.ollama,
            opensearch_client=self.opensearch,
            embeddings_client=self.embeddings,
            langfuse_tracer=self.langfuse_tracer,
            trace=trace,
            langfuse_enabled=self.langfuse_tracer is not None and self.langfuse_tracer.enabled,
            model_name=model_to_use,
            temperature=self.graph_config.temperature,
            top_k=self.graph_config.top_k,
            max_retrieval_attempts=self.graph_config.max_retrieval_attempts,
            guardrail_threshold=self.graph_config.guardrail_threshold,
        )

        # thread_id scopes LangGraph's in-memory checkpointing to this request
        config = {"configurable": {"thread_id": f"user_{user_id}_{int(start_time)}"}}

        logger.info("Invoking LangGraph workflow")
        result = await self.graph.ainvoke(
            state_input,
            config=config,
            context=runtime_context,
        )

        execution_time = time.time() - start_time
        logger.info(f"Graph execution completed in {execution_time:.2f}s")

        # -- Extract structured results from final state --------------------
        answer = self._extract_answer(result)
        sources = self._extract_sources(result)
        retrieval_attempts = result.get("retrieval_attempts", 0)
        reasoning_steps = self._extract_reasoning_steps(result)
        guardrail_result = result.get("guardrail_result")

        # -- Close Langfuse trace with outcome -----------------------------
        if trace:
            try:
                trace.update(
                    output={
                        "answer": answer,
                        "sources_count": len(sources),
                        "retrieval_attempts": retrieval_attempts,
                        "execution_time": execution_time,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to update Langfuse trace: {e}")
            self.langfuse_tracer.flush()

        logger.info("=" * 70)
        logger.info("Agentic RAG request completed")
        logger.info(f"  Answer length:       {len(answer)} characters")
        logger.info(f"  Sources found:       {len(sources)}")
        logger.info(f"  Retrieval attempts:  {retrieval_attempts}")
        logger.info(f"  Execution time:      {execution_time:.2f}s")
        logger.info("=" * 70)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "reasoning_steps": reasoning_steps,
            "retrieval_attempts": retrieval_attempts,
            "rewritten_query": result.get("rewritten_query"),
            "execution_time": execution_time,
            "guardrail_score": guardrail_result.score if guardrail_result else None,
        }

    def _extract_answer(self, result: dict) -> str:
        """Read the final answer from the last message in state.

        What it does:
            Returns the content of the last message in state["messages"].
            Works regardless of which node wrote it — generate_answer appends
            an AIMessage; out_of_scope appends a rejection AIMessage.

        Why last message:
            Both terminal nodes (generate_answer, out_of_scope) append their
            response as the final AIMessage. Reading messages[-1] is correct
            for both paths without any node-name awareness.

        Args:
            result: Final state dict from graph.ainvoke

        Returns:
            Content string of the last message, or fallback if no messages
        """
        messages = result.get("messages", [])
        if not messages:
            return "No answer generated."

        final_message = messages[-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)

    def _extract_sources(self, result: dict) -> List[dict]:
        """Parse retrieved paper sources from ToolMessages in state.

        What it does:
            Calls extract_sources_from_tool_messages on the full message list.
            This parses every ToolMessage from the "retrieve_papers" tool,
            extracts metadata (arxiv_id, title, authors, url), and deduplicates
            by arxiv_id. Each SourceItem is converted to a dict via .to_dict().

        Why not state["relevant_sources"]:
            No node explicitly writes to state["relevant_sources"]. Retrieved
            papers live in ToolMessages added to state["messages"] by ToolNode.
            extract_sources_from_tool_messages is the correct extraction point.

        Args:
            result: Final state dict from graph.ainvoke

        Returns:
            List of source dicts ready for JSON serialisation. Empty list if
            no papers were retrieved or all parse attempts failed.
        """
        messages = result.get("messages", [])
        source_items = extract_sources_from_tool_messages(messages)
        return [item.to_dict() for item in source_items]

    def _extract_reasoning_steps(self, result: dict) -> List[str]:
        """Build human-readable reasoning steps from state fields.

        What it does:
            Reads guardrail_result, retrieval_attempts, grading_results, and
            rewritten_query from the final state and converts them into a
            chronological list of plain-English step descriptions.

        Why strings (not ReasoningStep objects):
            The API response needs simple strings for the reasoning_steps field.
            ReasoningStep objects are richer but the router converts them to
            strings anyway. Building strings here avoids an extra serialisation
            step in the router.

        Args:
            result: Final state dict from graph.ainvoke

        Returns:
            List of step description strings in execution order
        """
        steps: List[str] = []

        guardrail_result = result.get("guardrail_result")
        if guardrail_result:
            steps.append(
                f"Domain validation: scored {guardrail_result.score}/100 — {guardrail_result.reason}"
            )

        retrieval_attempts = result.get("retrieval_attempts", 0)
        if retrieval_attempts > 0:
            steps.append(f"Retrieval: {retrieval_attempts} attempt(s) made")

        grading_results = result.get("grading_results", [])
        if grading_results:
            relevant_count = sum(1 for g in grading_results if g.is_relevant)
            steps.append(
                f"Grading: {relevant_count}/{len(grading_results)} documents rated relevant"
            )

        if result.get("rewritten_query"):
            steps.append(f"Query rewritten: '{result['rewritten_query']}'")

        if retrieval_attempts > 0:
            steps.append("Answer generated from retrieved context")

        return steps

    def get_graph_visualization(self) -> bytes:
        """Return the compiled graph as a PNG image.

        What it does:
            Calls draw_mermaid_png() on the compiled graph to produce a
            visual diagram of all nodes and edges as PNG bytes.

        Why it is useful:
            Debugging graph topology — verify that edges are wired correctly,
            routing is reachable, and no nodes are isolated.

        Returns:
            PNG image bytes

        Raises:
            ImportError: If pygraphviz is not installed
        """
        try:
            png_bytes = self.graph.get_graph().draw_mermaid_png()
            logger.info(f"Generated PNG visualization ({len(png_bytes)} bytes)")
            return png_bytes
        except ImportError as e:
            raise ImportError(
                "Graph visualization requires pygraphviz. "
                "Install with: pip install pygraphviz"
            ) from e

    def get_graph_mermaid(self) -> str:
        """Return the compiled graph as a mermaid diagram string.

        What it does:
            Calls draw_mermaid() to produce a mermaid syntax string that can
            be rendered in markdown viewers or the Mermaid Live Editor.

        Returns:
            Mermaid diagram syntax string
        """
        mermaid_str = self.graph.get_graph().draw_mermaid()
        logger.info(f"Generated mermaid diagram ({len(mermaid_str)} characters)")
        return mermaid_str

    def get_graph_ascii(self) -> str:
        """Return an ASCII art representation of the compiled graph.

        What it does:
            Calls print_ascii() to produce a simple text diagram useful
            for quick terminal inspection without needing Graphviz.

        Returns:
            ASCII art string
        """
        ascii_str = self.graph.get_graph().print_ascii()
        logger.info("Generated ASCII graph representation")
        return ascii_str


