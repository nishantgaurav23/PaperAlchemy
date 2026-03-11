"""Citation-enforcing prompt templates for the RAG chain (S5.2).

System prompt instructs the LLM to always cite sources using [N] notation.
User prompt builder formats retrieved chunks with numbered metadata.
"""

from __future__ import annotations

from src.schemas.api.search import SearchHit

SYSTEM_PROMPT = """\
You are a research assistant that answers questions based ONLY on the provided paper excerpts.

Rules:
1. Cite sources using inline [N] notation (e.g., [1], [2]) for every claim.
2. Only cite papers from the provided context — never invent references.
3. At the end of your answer, include a **Sources:** section listing each cited paper:
   N. [Paper Title](https://arxiv.org/abs/ARXIV_ID) — Authors, Year
4. If the provided context does not contain enough information to answer, say so honestly.
5. Keep answers concise, well-structured, and focused on the question.
"""


def build_user_prompt(*, query: str, search_hits: list[SearchHit]) -> str:
    """Build the user prompt with numbered context chunks and the question.

    Each chunk is labeled with its paper title, authors, and arXiv ID
    so the LLM can reference them by number.
    """
    context_parts: list[str] = []

    for i, hit in enumerate(search_hits, start=1):
        title = hit.title or "Unknown Title"
        authors = ", ".join(hit.authors) if hit.authors else "Unknown Authors"
        arxiv_id = hit.arxiv_id or "unknown"
        text = hit.chunk_text or hit.abstract or ""

        context_parts.append(f"[{i}] Title: {title}\n    Authors: {authors}\n    arXiv ID: {arxiv_id}\n    Content: {text}")

    context_block = "\n\n".join(context_parts)

    return (
        f"Context (paper excerpts):\n"
        f"{'=' * 60}\n"
        f"{context_block}\n"
        f"{'=' * 60}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using ONLY the context above. Cite sources with [N] and include a Sources section at the end."
    )
