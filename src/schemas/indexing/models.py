"""
Pydantic models for text chunking and indexing.

Why it's needed:
    When we split a paper into chunks, we need structured objects that carry                                                     
    both the text and its positional metadata through the pipeline. Without                                                      
    these models, we'd pass around raw dicts or tuples — losing type safety,                                                     
    validation, and IDE autocompletion. Pydantic models ensure every chunk                                                       
    has the required fields with correct types before it reaches OpenSearch.

What it does:
    - ChunkMetadata: Captures WHERE a chunk lives within the original paper.                                                     
      Tracks its position (chunk_index), character offsets (start_char/end_char),                                                
      size (word_count), overlap with neighbors, and which section it came from.                                                 
    - TextChunk: Pairs the actual chunk text with its metadata and links it                                                      
      back to the source paper via arxiv_id and paper_id.  .

How it helps:
    The TextChunker produces List[TextChunk], which the HybridIndexer consumes.
    Pydantic validation ensures every chunk has all required fields before
    we attempt to generate embeddings or index into OpenSearch. This catches
    bugs like missing arxiv_id at chunk creation time, not at indexing time.

    - chunk_index: Preserves reading order so chunks can be reassembled                                                          
    - start_char/end_char: Enables highlighting the exact source passage in the UI                                               
    - overlap_with_previous/next: Lets downstream code know how much text is                                                     
      shared between adjacent chunks (useful for deduplication in search results)                                                
    - section_title: Powers section-filtered search ("show me only Methods chunks")                                              
    - arxiv_id + paper_id: Links chunks back to their parent paper in PostgreSQL                                                 
      and the papers OpenSearch index 

    PDF text                                                                                                                         
            ↓                                                                                                                             
    TextChunker.chunk_text()                                                                                                         
        ↓                                                                                                                             
    List[TextChunk]  ←── each has .text + .metadata + .arxiv_id + .paper_id                                                          
        ↓                                                                                                                             
    HybridIndexer: sends chunk.text → Jina API → gets embedding vector                                                               
        ↓                                                                                                                             
    HybridIndexer: builds OpenSearch document from TextChunk fields + embedding                                                      
        ↓                                                                                                                             
    OpenSearch bulk index                                                                                                            
                                                                                                                                    
    Key design decisions:                                                                                                            
    - Pydantic BaseModel (not dataclass) — gives us .model_dump() for easy serialization to OpenSearch documents, plus automatic     
      validation.                                                                                                                      
    - overlap_with_previous/next — tracked per-chunk so we can deduplicate overlapping content when displaying consecutive search    
      results.                                                                                                                         
    - section_title is Optional — not all papers have parseable section headers; plain text papers get None.                         
    - Both arxiv_id and paper_id — arxiv_id is the human-readable identifier for the papers index; paper_id is the PostgreSQL UUID   
      for joining with the relational database. 

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

    chunk_index: int                        # 0-based position in the paper 
    start_char: int                         # Character offset where chunk starts in full text
    end_char: int                           # Character offset where chunk ends in full text
    word_count: int                         # Number of words in this chunk
    overlap_with_previous: int              # Words shared with the previous chunk 
    overlap_with_next: int                  # Words shared with the next chunk 
    section_title: Optional[str] = None     # Paper section (e.g., "Introduction", "Methods")


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

    text: str                   # The actual chunk text (target ~600 words) 
    metadata: ChunkMetadata     # Positional and structural metadata
    arxiv_id: str               # arXiv identifier (e.g., "2401.12345") — links to papers index  
    paper_id: str               # Internal UUID from PostgreSQL papers table
