"""Parent-child chunk retrieval for improved RAG precision.

Two-tier chunking strategy:
- Parent chunks (600 words): provide rich LLM context
- Child chunks (200 words): indexed for precise retrieval
- At query time, child matches are expanded to their parent chunks

Pipeline: Paper → ParentChildChunker → ParentChildResult → index children → retrieve → expand to parents
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.schemas.indexing import ChildChunk, ChunkMetadata, ParentChildResult, TextChunk
from src.services.indexing.text_chunker import TextChunker

logger = logging.getLogger(__name__)


class ParentChildChunker:
    """Two-tier chunker: small children for retrieval, large parents for context."""

    def __init__(
        self,
        parent_chunk_size: int = 600,
        child_chunk_size: int = 200,
        child_overlap: int = 50,
        min_chunk_size: int = 50,
    ):
        if child_chunk_size >= parent_chunk_size:
            raise ValueError("child_chunk_size must be less than parent_chunk_size")
        if child_overlap >= child_chunk_size:
            raise ValueError("child_overlap must be less than child_chunk_size")

        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.min_chunk_size = min_chunk_size

        # Reuse TextChunker for parent-level chunking
        self._parent_chunker = TextChunker(
            chunk_size=parent_chunk_size,
            overlap_size=100,
            min_chunk_size=min_chunk_size,
        )

        logger.info(
            "ParentChildChunker initialized: parent=%d, child=%d, overlap=%d",
            parent_chunk_size,
            child_chunk_size,
            child_overlap,
        )

    # ── FR-2: Create Parent-Child Chunks ─────────────────────────────

    def create_parent_child_chunks(
        self,
        title: str,
        abstract: str,
        full_text: str,
        arxiv_id: str,
        paper_id: str,
        sections: dict[str, str] | str | list | None = None,
    ) -> ParentChildResult:
        """Create parent chunks and split each into child sub-chunks."""
        if not full_text or not full_text.strip():
            logger.warning("Empty text for paper %s", arxiv_id)
            return ParentChildResult()

        # Create parent chunks using existing TextChunker
        parents = self._parent_chunker.chunk_paper(
            title=title,
            abstract=abstract,
            full_text=full_text,
            arxiv_id=arxiv_id,
            paper_id=paper_id,
            sections=sections,
        )

        if not parents:
            return ParentChildResult()

        # Split each parent into children
        all_children: list[ChildChunk] = []
        for parent in parents:
            children = self.split_parent_into_children(parent)
            all_children.extend(children)

        logger.info(
            "Paper %s: %d parents → %d children",
            arxiv_id,
            len(parents),
            len(all_children),
        )

        return ParentChildResult(parents=parents, children=all_children)

    # ── FR-4: Split Parent into Children ─────────────────────────────

    def split_parent_into_children(self, parent: TextChunk) -> list[ChildChunk]:
        """Split a parent chunk into overlapping child chunks."""
        text = parent.text
        if not text or not text.strip():
            return []

        words = re.findall(r"\S+", text)

        if len(words) <= self.child_chunk_size:
            # Single child covering the entire parent
            child = ChildChunk(
                text=text,
                metadata=ChunkMetadata(
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text),
                    word_count=len(words),
                    overlap_with_previous=0,
                    overlap_with_next=0,
                    section_title=parent.metadata.section_title,
                ),
                arxiv_id=parent.arxiv_id,
                paper_id=parent.paper_id,
                parent_chunk_index=parent.metadata.chunk_index,
            )
            return [child]

        children: list[ChildChunk] = []
        step = self.child_chunk_size - self.child_overlap
        child_index = 0
        pos = 0

        while pos < len(words):
            end = min(pos + self.child_chunk_size, len(words))
            chunk_words = words[pos:end]
            chunk_text = " ".join(chunk_words)

            overlap_prev = min(self.child_overlap, pos) if pos > 0 else 0
            overlap_next = self.child_overlap if end < len(words) else 0

            child = ChildChunk(
                text=chunk_text,
                metadata=ChunkMetadata(
                    chunk_index=child_index,
                    start_char=len(" ".join(words[:pos])) if pos > 0 else 0,
                    end_char=len(" ".join(words[:end])),
                    word_count=len(chunk_words),
                    overlap_with_previous=overlap_prev,
                    overlap_with_next=overlap_next,
                    section_title=parent.metadata.section_title,
                ),
                arxiv_id=parent.arxiv_id,
                paper_id=parent.paper_id,
                parent_chunk_index=parent.metadata.chunk_index,
            )
            children.append(child)

            pos += step
            child_index += 1

            if end >= len(words):
                break

        return children

    # ── FR-5: Parent Expansion (Retrieval-Time) ──────────────────────

    def expand_to_parents(
        self,
        child_results: list[dict[str, Any]],
        os_client: Any,
    ) -> list[dict[str, Any]]:
        """Expand child search results to their parent chunks, deduplicating."""
        if not child_results:
            return []

        # Group by parent_chunk_id, keeping best score
        parent_map: dict[str, dict[str, Any]] = {}
        orphans: list[dict[str, Any]] = []

        for child in child_results:
            parent_id = child.get("parent_chunk_id")
            if not parent_id:
                orphans.append(child)
                continue

            if parent_id not in parent_map or child.get("score", 0) > parent_map[parent_id].get("score", 0):
                parent_map[parent_id] = {"parent_chunk_id": parent_id, "score": child.get("score", 0), "child": child}

        # Fetch parent documents from OpenSearch
        results: list[dict[str, Any]] = []
        for parent_id, entry in parent_map.items():
            try:
                doc = os_client.client.get(index=os_client.index_name, id=parent_id)
                parent_data = doc["_source"]
                parent_data["score"] = entry["score"]
                parent_data["expanded_from_child"] = True
                results.append(parent_data)
            except Exception:
                logger.warning("Failed to fetch parent %s, returning child as-is", parent_id)
                results.append(entry["child"])

        results.extend(orphans)
        return results

    # ── FR-6: Prepare Children for Indexing ───────────────────────────

    def prepare_for_indexing(self, result: ParentChildResult) -> list[dict[str, Any]]:
        """Convert ParentChildResult children into OpenSearch-ready dicts."""
        if not result.children:
            return []

        # Build parent lookup for metadata
        parent_lookup: dict[int, TextChunk] = {p.metadata.chunk_index: p for p in result.parents}

        docs: list[dict[str, Any]] = []
        for child in result.children:
            parent = parent_lookup.get(child.parent_chunk_index)
            parent_chunk_id = f"{child.arxiv_id}_parent_{child.parent_chunk_index}"

            chunk_data: dict[str, Any] = {
                "chunk_text": child.text,
                "arxiv_id": child.arxiv_id,
                "paper_id": child.paper_id,
                "chunk_index": child.metadata.chunk_index,
                "chunk_word_count": child.metadata.word_count,
                "start_char": child.metadata.start_char,
                "end_char": child.metadata.end_char,
                "parent_chunk_id": parent_chunk_id,
                "section_title": child.metadata.section_title,
            }

            # Copy parent-level metadata if available
            if parent and parent.metadata.section_title:
                chunk_data["section_title"] = parent.metadata.section_title

            docs.append({"chunk_data": chunk_data})

        logger.info("Prepared %d child chunks for indexing", len(docs))
        return docs
