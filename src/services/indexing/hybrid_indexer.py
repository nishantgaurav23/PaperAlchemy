"""
Hybrid Indexing Service — orchestrates chunk → embed → index pipeline.

Why it's needed:
    Indexing a paper for hybrid search requires three coordinated steps:
    1. Chunk the document into searchable segments
    2. Generate embeddings for each chunk
    3. Store chunks + embeddings in OpenSearch
    Without an orchestrator, this logic would be scattered across notebooks,
    routers, and scripts — leading to inconsistency and duplication.

What it does:
    - index_paper(): Full pipeline for one paper. Chunks it, embeds all
      chunks in batches, indexes into OpenSearch. Returns statistics.
    - index_papers_batch(): Processes multiple papers sequentially.
      Optionally deletes existing chunks before re-indexing.
    - reindex_paper(): Deletes old chunks then re-indexes. Used when
      a paper's content is updated (e.g., new PDF parse).

How it helps:
    - Single responsibility: one class owns the entire indexing flow
    - Error isolation: failures in one paper don't stop the batch
    - Statistics tracking: returns counts of chunks created/indexed/failed
    - Denormalization: copies paper metadata into each chunk document
      so OpenSearch can return complete search results without database joins

Architecture:
    HybridIndexingService composes three dependencies:
    - TextChunker: splits papers into chunks
    - JinaEmbeddingsClient: generates vectors (async)
    - OpenSearchClient: stores chunks + vectors

    Created by factory.py, which wires up all dependencies from settings.

    Pipeline:
        Paper dict → TextChunker.chunk_paper()
            → List[TextChunk]
                → JinaClient.embed_passages(chunk texts)
                    → List[embedding]
                        → OpenSearchClient.bulk_index_chunks()
                            → OpenSearch index
"""

import logging
from typing import Dict, List

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient
from .text_chunker import TextChunker

logger = logging.getLogger(__name__)


class HybridIndexingService:
    """Orchestrates paper chunking, embedding, and OpenSearch indexing.

    This service is the central coordinator for the indexing pipeline.
    It does NOT own its dependencies — they are injected via __init__,
    making this class easy to test with mocks.
    """

    def __init__(
        self,
        chunker: TextChunker,
        embeddings_client: JinaEmbeddingsClient,
        opensearch_client: OpenSearchClient
    ):
        """Initialize hybrid indexing service with its dependencies.

        Args:
            chunker: TextChunker instance for splitting papers into chunks.
                     Configured with chunk_size, overlap_size from settings.
            embeddings_client: JinaEmbeddingsClient for generating vectors.
                               Must be an async client (uses await internally).
            opensearch_client: OpenSearchClient for storing indexed chunks.
                               Must have the hybrid index already created.
        """
        self.chunker = chunker
        self.embeddings_client = embeddings_client
        self.opensearch_client = opensearch_client

        logger.info("Hybrid indexing service initialized")

    async def index_paper(self, paper_data: Dict) -> Dict[str, int]:
        """Index a single paper: chunk → embed → store in OpenSearch.

        This is the core pipeline method. It processes one paper through
        the full indexing flow and returns statistics about what happened.

        Args:
            paper_data: Dictionary with paper fields from the database:
                - arxiv_id (str): Required. ArXiv identifier.
                - id (int): Database primary key.
                - title (str): Paper title.
                - abstract (str): Paper abstract.
                - raw_text/full_text (str): Full paper content from PDF.
                - sections (dict/list/str): Parsed sections from PDF.
                - authors (list/str): Author names.
                - categories (list): arXiv category codes.
                - published_date (str): Publication date ISO string.

        Returns:
            Dictionary with indexing statistics:
                - chunks_created: Number of chunks produced by TextChunker
                - chunks_indexed: Number successfully stored in OpenSearch
                - embeddings_generated: Number of embeddings from Jina
                - errors: Number of failures (0 = complete success)
        """
        arxiv_id = paper_data.get("arxiv_id")
        paper_id = str(paper_data.get("id", ""))

        if not arxiv_id:
            logger.error("Paper missing arxiv_id")
            return {
                "chunks_created": 0,
                "chunks_indexed": 0,
                "embeddings_generated": 0,
                "errors": 1
            }

        try:
            # ─── Step 1: Chunk the paper ───────────────────────────
            # Uses section-based chunking if sections are available,
            # falls back to word-based sliding window otherwise.
            chunks = self.chunker.chunk_paper(
                title=paper_data.get("title", ""),
                abstract=paper_data.get("abstract", ""),
                full_text=paper_data.get(
                    "raw_text", paper_data.get("full_text", "")
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
                sections=paper_data.get("sections"),
            )

            if not chunks:
                logger.warning(f"No chunks created for paper {arxiv_id}")
                return {
                    "chunks_created": 0,
                    "chunks_indexed": 0,
                    "embeddings_generated": 0,
                    "errors": 0
                }

            logger.info(
                f"Created {len(chunks)} chunks for paper {arxiv_id}"
            )

            # ─── Step 2: Generate embeddings ───────────────────────
            # Embeds all chunk texts in batches of 50.
            # Uses task="retrieval.passage" for document-side encoding.
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.embeddings_client.embed_passages(
                texts=chunk_texts,
                batch_size=50,  # Process 50 chunks per API call
            )

            # Verify embedding count matches chunk count
            if len(embeddings) != len(chunks):
                logger.error(
                    f"Embedding count mismatch: "
                    f"{len(embeddings)} != {len(chunks)}"
                )
                return {
                    "chunks_created": len(chunks),
                    "chunks_indexed": 0,
                    "embeddings_generated": len(embeddings),
                    "errors": 1
                }

            # ─── Step 3: Prepare documents for OpenSearch ──────────
            # Each document contains:
            # - Chunk-specific data (text, position, embedding)
            # - Denormalized paper metadata (title, authors, etc.)
            #   so search results don't require a database join
            chunks_with_embeddings = []

            for chunk, embedding in zip(chunks, embeddings):
                # Build the OpenSearch document
                chunk_data = {
                    # Chunk-specific fields
                    "arxiv_id": chunk.arxiv_id,
                    "paper_id": chunk.paper_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "chunk_text": chunk.text,
                    "chunk_word_count": chunk.metadata.word_count,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "section_title": chunk.metadata.section_title,
                    "embedding_model": "jina-embeddings-v3",
                    # Denormalized paper metadata for search results
                    # This avoids a database round-trip when displaying
                    # search results to the user
                    "title": paper_data.get("title", ""),
                    "authors": (
                        ", ".join(paper_data.get("authors", []))
                        if isinstance(paper_data.get("authors"), list)
                        else paper_data.get("authors", "")
                    ),
                    "abstract": paper_data.get("abstract", ""),
                    "categories": paper_data.get("categories", []),
                    "published_date": paper_data.get("published_date"),
                }

                chunks_with_embeddings.append({
                    "chunk_data": chunk_data,
                    "embedding": embedding
                })

            # ─── Step 4: Bulk index into OpenSearch ────────────────
            # Uses bulk API for efficiency (single request for all chunks)
            results = self.opensearch_client.bulk_index_chunks(
                chunks_with_embeddings
            )

            logger.info(
                f"Indexed paper {arxiv_id}: "
                f"{results['success']} chunks successful, "
                f"{results['failed']} failed"
            )

            return {
                "chunks_created": len(chunks),
                "chunks_indexed": results["success"],
                "embeddings_generated": len(embeddings),
                "errors": results["failed"],
            }

        except Exception as e:
            logger.error(f"Error indexing paper {arxiv_id}: {e}")
            return {
                "chunks_created": 0,
                "chunks_indexed": 0,
                "embeddings_generated": 0,
                "errors": 1
            }

    async def index_papers_batch(
        self,
        papers: List[Dict],
        replace_existing: bool = False
    ) -> Dict[str, int]:
        """Index multiple papers sequentially.

        Processes papers one at a time to avoid overwhelming the Jina API
        with too many concurrent requests. Each paper failure is isolated
        — one bad paper doesn't stop the batch.

        Args:
            papers: List of paper data dictionaries (see index_paper docs).
            replace_existing: If True, delete existing chunks for each paper
                              before re-indexing. Useful for re-processing
                              after PDF parser improvements.

        Returns:
            Aggregated statistics across all papers:
                - papers_processed: Total papers attempted
                - total_chunks_created: Sum of chunks from TextChunker
                - total_chunks_indexed: Sum of chunks stored in OpenSearch
                - total_embeddings_generated: Sum of Jina API embeddings
                - total_errors: Sum of failures across all papers
        """
        # Accumulate statistics across all papers
        total_stats = {
            "papers_processed": 0,
            "total_chunks_created": 0,
            "total_chunks_indexed": 0,
            "total_embeddings_generated": 0,
            "total_errors": 0,
        }

        for paper in papers:
            arxiv_id = paper.get("arxiv_id")

            # Optionally clear existing chunks before re-indexing
            if replace_existing and arxiv_id:
                self.opensearch_client.delete_paper_chunks(arxiv_id)

            # Index this paper (errors are caught internally)
            stats = await self.index_paper(paper)

            # Aggregate statistics
            total_stats["papers_processed"] += 1
            total_stats["total_chunks_created"] += stats["chunks_created"]
            total_stats["total_chunks_indexed"] += stats["chunks_indexed"]
            total_stats["total_embeddings_generated"] += stats[
                "embeddings_generated"
            ]
            total_stats["total_errors"] += stats["errors"]

        logger.info(
            f"Batch indexing complete: "
            f"{total_stats['papers_processed']} papers, "
            f"{total_stats['total_chunks_indexed']} chunks indexed"
        )

        return total_stats

    async def reindex_paper(
        self,
        arxiv_id: str,
        paper_data: Dict
    ) -> Dict[str, int]:
        """Delete old chunks and re-index a paper with fresh data.

        Used when a paper's content has been updated (e.g., better PDF
        parse, corrected metadata). Ensures no stale chunks remain.

        Args:
            arxiv_id: ArXiv ID of the paper to reindex.
            paper_data: Updated paper data dictionary.

        Returns:
            Indexing statistics (same as index_paper).
        """
        # Delete all existing chunks for this paper
        deleted = self.opensearch_client.delete_paper_chunks(arxiv_id)
        if deleted:
            logger.info(f"Deleted existing chunks for paper {arxiv_id}")

        # Re-index with fresh data
        return await self.index_paper(paper_data)
