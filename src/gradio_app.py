"""
What is needed:
    A Gradio web interface that exposes PaperAlchemy's FastAPI endpoints
    as a browser-based chat UI. No new backend logic — purely a client
    that calls the already-running API over HTTP.

Why it is needed:
    The only way to query PaperAlchemy today is curl or /docs (Swagger).
    Gradio provides a shareable, visual interface in one file with zero
    additional infrastructure. It demonstrates both pipelines side by side:
    - Standard RAG: streaming tokens, keyword/hybrid toggle, category filter
    - Agentic RAG: full reasoning trace, rich source cards, guardrail score

How it helps:
    - Two tabs: one per pipeline — users pick based on their need
    - Streaming tab shows the answer building token-by-token (better UX)
    - Agentic tab reveals every decision the agent made (transparency)
    - Examples panel lets new users explore without typing
    - Out-of-scope example ("What is 2+2?") demonstrates the guardrail live

____________________________________________________________________________________________________________
What it does:
    Defines two async handler functions + a Gradio Blocks layout:

┌────────────────────────────┬────────────────────────────────────────────────────────────┐
│          Function          │                         Purpose                            │
├────────────────────────────┼────────────────────────────────────────────────────────────┤
│ stream_rag_response        │ Async generator — streams /stream SSE tokens to Gradio     │
├────────────────────────────┼────────────────────────────────────────────────────────────┤
│ _format_stream_output      │ Pure helper — appends source/search-info footer to answer  │
├────────────────────────────┼────────────────────────────────────────────────────────────┤
│ ask_agentic                │ Async function — POSTs to /ask-agentic, formats response   │
├────────────────────────────┼────────────────────────────────────────────────────────────┤
│ _format_agentic_output     │ Pure helper — converts agentic dict → rich Markdown string │
├────────────────────────────┼────────────────────────────────────────────────────────────┤
│ create_gradio_interface    │ Builds the Gradio Blocks UI with two tabs                  │
├────────────────────────────┼────────────────────────────────────────────────────────────┤
│ main                       │ Entry point — compiles UI and calls interface.launch()     │
└────────────────────────────┴────────────────────────────────────────────────────────────┘

Why two tabs (not two separate apps):
    gr.Blocks with gr.Tabs keeps both pipelines in one URL and one process.
    Users can switch between standard and agentic without navigating away.

Why async handlers:
    Gradio natively supports async generator functions for streaming.
    stream_rag_response must be async to use httpx.AsyncClient.stream().
    ask_agentic is async to use await client.post() without blocking the
    Gradio event loop.

Why httpx (not requests):
    httpx has a native async API (AsyncClient) and supports streaming
    (client.stream()). requests is synchronous-only — it would block
    Gradio's async event loop during long LLM generations.

Why sources differ between tabs:
    Standard RAG (/stream) returns sources as List[str] (PDF URLs).
    Agentic RAG (/ask-agentic) returns sources as List[dict] with
    arxiv_id, title, authors, url, relevance_score. The two _format_*
    helpers handle each shape separately.

Timeout differences:
    /stream timeout=120s  — streaming starts fast, tokens arrive quickly
    /ask-agentic timeout=180s — full pipeline (multiple LLM calls) may need more

"""
import json
import logging
from typing import AsyncGenerator

import gradio as gr
import httpx

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000/api/v1"
DEFAULT_MODEL = "llama3.2:1b"

async def stream_rag_response(
    query: str,
    top_k: int = 5,
    use_hybrid: bool = True,
    model: str = DEFAULT_MODEL,
    categories: str = "",
) -> AsyncGenerator[str, None]:
    """Stream tokens from the standard RAG /stream endpoint.

    What it does:
        Opens a persistent HTTP connection to /stream, reads Server-Sent
        Events line by line, accumulates the answer, and yields the
        growing formatted string to Gradio on every token chunk.

    Why a generator (not return):
        Gradio renders each yielded value immediately, giving the user
        a live view of the answer being written. A plain return would
        wait for the full response — no streaming UX benefit.

    Args:
        query:       User question (empty → early return with prompt)
        top_k:       Number of chunks to retrieve (1-10)
        use_hybrid:  BM25 + vector search if True, BM25-only if False
        model:       Ollama model name
        categories:  Comma-separated arXiv category filter (optional)

    Yields:
        Incrementally-built Markdown string as tokens arrive
    """
    if not query.strip():
        yield "Please enter a question."
        return

    category_list = (
        [c.strip() for c in categories.split(",") if c.strip()] if categories else None
    )
    payload = {
        "query": query,
        "top_k": top_k,
        "use_hybrid": use_hybrid,
        "model": model,
        "categories": category_list,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{API_BASE_URL}/stream",
                json=payload,
                headers={"Accept": "text/plain"},
            ) as response:
                if response.status_code != 200:
                    yield f"Error: API returned status {response.status_code}"
                    return

                current_answer = ""
                sources: list = []
                chunks_used = 0
                search_mode = ""

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        yield f"Error: {data['error']}"
                        return

                    # 1. Done events take priority (may also carry sources + answer)
                    if data.get("done", False):
                        if "sources" in data:
                            sources = data.get("sources", [])
                            chunks_used = data.get("chunks_used", 0)
                            search_mode = data.get("search_mode", "")
                        final = data.get("answer", current_answer)
                        yield _format_stream_output(final, sources, chunks_used, search_mode)
                        break

                    # 2. Cache hit: has "answer" but no "done" key (AskResponse dump)
                    if "answer" in data and "chunk" not in data:
                        if "sources" in data:
                            sources = data.get("sources", [])
                            chunks_used = data.get("chunks_used", 0)
                            search_mode = data.get("search_mode", "")
                        yield _format_stream_output(data["answer"], sources, chunks_used, search_mode)
                        break

                    # 3. Pure metadata event (sources only, no answer/done)
                    if "sources" in data:
                        sources = data["sources"]
                        chunks_used = data.get("chunks_used", 0)
                        search_mode = data.get("search_mode", "")
                        continue

                    # 4. Streaming token
                    if "chunk" in data:
                        current_answer += data["chunk"]
                        yield _format_stream_output(
                            current_answer, sources, chunks_used, search_mode
                        )

    except httpx.RequestError as e:
        yield (
            f"Connection error: {e}\n\n"
            f"Make sure the PaperAlchemy API is running at `{API_BASE_URL}`."
        )
    except Exception as e:
        yield f"Unexpected error: {e}"


def _format_stream_output(
    answer: str, sources: list, chunks_used: int, search_mode: str
) -> str:
    """Append search metadata footer to the growing answer.

    What it does:
        Builds a Markdown string: answer text + optional source URLs and
        search-mode info. Called on every token chunk so the footer is
        visible as soon as metadata arrives.

    Args:
        answer:      Current accumulated answer text
        sources:     List of PDF URL strings (may be empty early in stream)
        chunks_used: Number of chunks used as context
        search_mode: "hybrid" or "bm25"

    Returns:
        Full Markdown string ready for Gradio Markdown component
    """
    output = answer

    if sources or chunks_used:
        output += f"\n\n---\n**Search:** mode=`{search_mode}`, chunks=`{chunks_used}`\n"

    if sources:
        output += f"\n**Sources ({len(sources)} papers):**\n"
        for i, src in enumerate(sources[:5], 1):
            label = src.split("/")[-1] if isinstance(src, str) else str(src)
            output += f"{i}. [{label}]({src})\n"
        if len(sources) > 5:
            output += f"...and {len(sources) - 5} more\n"

    return output


async def ask_agentic(query: str, model: str = DEFAULT_MODEL) -> str:
    """Call the agentic RAG /ask-agentic endpoint and return formatted Markdown.

    What it does:
        POSTs to /ask-agentic with the query and optional model override.
        Waits for the full response (the agentic pipeline is not streaming),
        then calls _format_agentic_output to convert the rich dict into
        a readable Markdown string.

    Why not streaming:
        The agentic pipeline runs multiple sequential LLM calls (guardrail,
        grading, generation). Streaming mid-pipeline would require
        restructuring the LangGraph workflow — a future enhancement.

    Args:
        query: User question
        model: Optional Ollama model override

    Returns:
        Formatted Markdown string with answer, reasoning, and sources
    """
    if not query.strip():
        return "Please enter a question."

    payload = {"query": query, "model": model or None}

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(f"{API_BASE_URL}/ask-agentic", json=payload)

        if response.status_code != 200:
            return f"Error {response.status_code}: {response.text}"

        return _format_agentic_output(response.json())

    except httpx.RequestError as e:
        return (
            f"Connection error: {e}\n\n"
            f"Make sure the PaperAlchemy API is running at `{API_BASE_URL}`."
        )
    except Exception as e:
        return f"Unexpected error: {e}"


def _format_agentic_output(data: dict) -> str:
    """Convert the /ask-agentic response dict to a rich Markdown string.

    What it does:
        Renders: guardrail score → rewritten query (if any) → answer →
        numbered reasoning steps → source cards with title/authors/arxiv_id
        → metadata footer (attempts, model, execution time).

    Why separate from ask_agentic:
        Pure function — easy to unit test without an HTTP call.
        ask_agentic handles I/O; this handles formatting.

    Args:
        data: Parsed JSON dict from /ask-agentic response

    Returns:
        Markdown string for the Gradio Markdown component
    """
    output = ""

    # Guardrail score
    score = data.get("guardrail_score")
    if score is not None:
        bar = "🟢" if score >= 70 else ("🟡" if score >= 40 else "🔴")
        output += f"{bar} **Domain relevance:** {score}/100\n\n"

    # Rewritten query (only shown if the agent rewrote it)
    if data.get("rewritten_query"):
        output += f"✏️  **Query rewritten to:** _{data['rewritten_query']}_\n\n"

    # Main answer
    output += f"### Answer\n\n{data.get('answer', 'No answer generated.')}\n\n"

    # Reasoning steps
    steps = data.get("reasoning_steps", [])
    if steps:
        output += "---\n### Agent Reasoning Steps\n\n"
        for i, step in enumerate(steps, 1):
            output += f"{i}. {step}\n"
        output += "\n"

    # Source cards
    sources = data.get("sources", [])
    if sources:
        output += f"---\n### Sources ({len(sources)} papers)\n\n"
        for src in sources:
            title = src.get("title") or "Unknown title"
            url = src.get("url", "#")
            arxiv_id = src.get("arxiv_id", "")
            authors = src.get("authors", [])
            score_val = src.get("relevance_score", 0.0)
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            output += f"- **[{title}]({url})**\n"
            output += (
                f"  `arXiv:{arxiv_id}` | {author_str or 'Unknown'} "
                f"| relevance: `{score_val:.2f}`\n\n"
            )

    # Footer metadata
    attempts = data.get("retrieval_attempts", 0)
    exec_time = data.get("execution_time", 0.0)
    model = data.get("model", "")
    output += (
        f"---\n*Retrieval attempts: {attempts} "
        f"| Model: `{model}` "
        f"| Time: {exec_time:.2f}s*"
    )
    return output


def create_gradio_interface() -> gr.Blocks:
    """Build the Gradio Blocks UI with Standard RAG and Agentic RAG tabs.

    What it does:
        Creates a gr.Blocks layout with:
        - Header with PaperAlchemy branding
        - Tab 1: Standard RAG with streaming, top_k slider, hybrid toggle,
                model dropdown, category filter, and example queries
        - Tab 2: Agentic RAG with model override and example queries
                including an out-of-scope demo ("What is 2+2?")
        - Footer with usage notes

    Returns:
        Compiled gr.Blocks interface ready for .launch()
    """
    with gr.Blocks(
        title="PaperAlchemy — Research Paper RAG",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown(
            """
            # PaperAlchemy
            **Transform Academic Papers into Knowledge Gold**

            Ask questions about CS, AI, and ML research papers indexed from arXiv.
            Choose between **Standard RAG** (streaming) or the **Agentic pipeline**
            (intelligent retrieval with reasoning transparency).
            """
        )

        with gr.Tabs():

            # ── Tab 1: Standard RAG ──────────────────────────────────────────
            with gr.TabItem("Standard RAG (Streaming)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        rag_query = gr.Textbox(
                            label="Your Question",
                            placeholder="What are transformers in machine learning?",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        rag_btn = gr.Button("Ask", variant="primary", size="lg")

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        rag_top_k = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Chunks to retrieve",
                            info="More chunks = more context, slower generation",
                        )
                        rag_hybrid = gr.Checkbox(
                            value=True,
                            label="Hybrid search (BM25 + vector)",
                            info="Better recall than keyword-only",
                        )
                    with gr.Row():
                        rag_model = gr.Dropdown(
                            choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"],
                            value=DEFAULT_MODEL,
                            label="Model",
                        )
                        rag_categories = gr.Textbox(
                            label="arXiv categories (optional)",
                            placeholder="cs.AI, cs.LG",
                            info="Comma-separated. Leave empty for all.",
                        )

                rag_output = gr.Markdown(
                    value="Ask a question to get started!",
                    label="Answer",
                    height=400,
                )

                gr.Examples(
                    examples=[
                        ["What are transformers in machine learning?", 5, True, "llama3.2:1b", "cs.AI, cs.LG"],
                        ["How do diffusion models generate images?", 5, True, "llama3.2:1b", "cs.CV"],
                        ["Explain RLHF in large language models", 3, True, "llama3.2:1b", "cs.AI"],
                        ["What is retrieval-augmented generation?", 5, True, "llama3.2:1b", "cs.AI, cs.CL"],
                    ],
                    inputs=[rag_query, rag_top_k, rag_hybrid, rag_model, rag_categories],
                )

                rag_btn.click(
                    fn=stream_rag_response,
                    inputs=[rag_query, rag_top_k, rag_hybrid, rag_model, rag_categories],
                    outputs=rag_output,
                )
                rag_query.submit(
                    fn=stream_rag_response,
                    inputs=[rag_query, rag_top_k, rag_hybrid, rag_model, rag_categories],
                    outputs=rag_output,
                )

            # ── Tab 2: Agentic RAG ───────────────────────────────────────────
            with gr.TabItem("Agentic RAG (LangGraph)"):
                gr.Markdown(
                    """
                    The **agentic pipeline** runs a LangGraph workflow:
                    `guardrail → retrieve → grade documents → (rewrite if needed) → generate`

                    Every decision is explained in the **Agent Reasoning Steps** section below the answer.
                    Try the out-of-scope example to see the domain guardrail in action.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        ag_query = gr.Textbox(
                            label="Your Question",
                            placeholder="How does BERT's pre-training differ from GPT?",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        ag_btn = gr.Button("Ask Agent", variant="primary", size="lg")

                ag_model = gr.Dropdown(
                    choices=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"],
                    value=DEFAULT_MODEL,
                    label="Model override (optional)",
                )

                ag_output = gr.Markdown(
                    value="Ask a question to see the agent's reasoning process!",
                    label="Answer + Reasoning",
                    height=500,
                )

                gr.Examples(
                    examples=[
                        ["How does BERT's pre-training differ from GPT?", "llama3.2:1b"],
                        ["What are the key ideas behind LoRA fine-tuning?", "llama3.2:1b"],
                        ["Explain the architecture of Vision Transformers (ViT)", "llama3.2:1b"],
                        ["What is 2 + 2?", "llama3.2:1b"],  # Out-of-scope guardrail demo
                    ],
                    inputs=[ag_query, ag_model],
                )

                ag_btn.click(fn=ask_agentic, inputs=[ag_query, ag_model], outputs=ag_output)
                ag_query.submit(fn=ask_agentic, inputs=[ag_query, ag_model], outputs=ag_output)

            # ── Tab 3: Infrastructure ────────────────────────────────────────
            with gr.TabItem("Infrastructure"):
                gr.Markdown(
                    """
                    ## Service Links

                    Click any link to open the service dashboard in a new tab.

                    | Service | URL | Credentials |
                    |---|---|---|
                    | **API Swagger Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | — |
                    | **Gradio UI** | [http://localhost:7861](http://localhost:7861) | — (this page) |
                    | **Airflow** | [http://localhost:8080](http://localhost:8080) | `airflow` / `airflow` |
                    | **pgAdmin** | [http://localhost:5050](http://localhost:5050) | `admin@paperalchemy.dev` / `paperalchemy_secret` |
                    | **OpenSearch Dashboards** | [http://localhost:5602](http://localhost:5602) | — |
                    | **Langfuse** | [http://localhost:3001](http://localhost:3001) | — |
                    | **MinIO Console** | [http://localhost:9091](http://localhost:9091) | — |
                    """
                )

        gr.Markdown(
            """
            ---
            Ensure the PaperAlchemy API is running at `http://localhost:8000` before using this interface.

            **arXiv categories:** `cs.AI` · `cs.LG` · `cs.CL` · `cs.CV` · `cs.NE` · `stat.ML`
            """
        )

    return interface


def main() -> None:
    """Launch the Gradio interface on port 7861."""
    print("Starting PaperAlchemy Gradio Interface...")
    print(f"API: {API_BASE_URL}")
    print("UI:  http://localhost:7861")

    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()