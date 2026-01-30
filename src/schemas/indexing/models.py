"""
Pydantic models for text chunking and indexing.

Why it's needed:
    When we split a paper into chunks for indexing, each chunk needs metadata
    (position, word count, section title, overlap info). Without structured
    models, this metadata would be scattered across dicts with inconsistent
    keys, making bugs hard to find and code hard to maintain.

What it does:
    - ChunkMetadata: Tracks where a chunk came from in the original document
      (character offsets, word count, overlap amounts, section title).
    - TextChunk: Pairs the actual chunk text with its metadata, plus the
      paper identifiers (arxiv_id, paper_id) needed for OpenSearch indexing.

How it helps:
    The TextChunker produces List[TextChunk], which the HybridIndexer consumes.
    Pydantic validation ensures every chunk has all required fields before
    we attempt to generate embeddings or index into OpenSearch. This catches
    bugs like missing arxiv_id at chunk creation time, not at indexing time.
"""

from typing import Optional

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata describing a chunk's position and context within a paper.

    This metadata is stored alongside the chunk text in OpenSearch, enabling:
    - Reconstructing the original reading order (chunk_index)
    - Highlighting the source location (start_char, end_char)
    - Filtering by section (section_title)
    - Debugging chunking quality (word_count, overlaps)

    Attributes:
        chunk_index: Zero-based position of this chunk within the paper.
                     Used to reconstruct reading order when displaying
                     multiple chunks from the same paper.
        start_char: Character offset where this chunk starts in the
                    original full text. Used for source highlighting.
        end_char: Character offset where this chunk ends. Together with
                  start_char, defines the exact span in the original.
        word_count: Number of words in this chunk. Used to verify
                    chunking quality (should be ~chunk_size from config).
        overlap_with_previous: Number of words shared with the previous
                               chunk. Ensures no information is lost at
                               chunk boundaries.
        overlap_with_next: Number of words shared with the next chunk.
                           Zero for the last chunk in a paper.
        section_title: Name of the paper section this chunk belongs to
                       (e.g., "Introduction", "Methods"). None if the
                       paper wasn't parsed with section information or
                       if traditional word-based chunking was used.
    """

    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    overlap_with_previous: int
    overlap_with_next: int
    section_title: Optional[str] = None


class TextChunk(BaseModel):
    """A chunk of text with its metadata and paper identifiers.

    This is the primary data structure flowing through the indexing pipeline:
        TextChunker.chunk_paper() → List[TextChunk]
        → HybridIndexer embeds each chunk.text
        → OpenSearch indexes chunk data + embedding vector

    Attributes:
        text: The actual text content of this chunk. Typically 400-800 words
              for section-based chunks, or chunk_size (default 600) words
              for traditional word-based chunks. This text is what gets
              embedded by Jina and searched by BM25.
        metadata: Positional and contextual metadata (see ChunkMetadata).
        arxiv_id: The arXiv identifier (e.g., "2401.12345") of the source
                  paper. Used as part of the OpenSearch document ID
                  (format: "{arxiv_id}_{chunk_index}") and for filtering
                  search results by paper.
        paper_id: The PostgreSQL primary key of the source paper. Stored
                  in OpenSearch for efficient joins back to the database
                  when additional paper metadata is needed.
    """

    text: str
    metadata: ChunkMetadata
    arxiv_id: str
    paper_id: str
