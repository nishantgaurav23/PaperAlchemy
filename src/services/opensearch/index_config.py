"""
OpenSearch index mappings, analyzers, and search pipeline configuration.

Why it's needed:
    OpenSearch needs explicit index mappings to know how to index and search
    each field. Without mappings, it guesses types (often wrong) and uses
    default analyzers (no stemming, no stop words). Strict dynamic mapping
    prevents accidental field creation that wastes disk and causes confusion.

What it does:
    - ARXIV_PAPERS_MAPPING: Simple paper-level index for BM25 keyword search.
      Uses text_analyzer (lowercase + stop words + snowball stemming) for
      title/abstract and standard_analyzer for authors. Strict mapping rejects
      documents with unmapped fields.

    - ARXIV_PAPERS_CHUNKS_MAPPING: Hybrid chunk-level index for BM25 + vector
      search. Adds knn_vector field (1024 dims, HNSW algorithm, cosine similarity)
      for semantic search alongside the same text analyzers for BM25.

    - HYBRID_RRF_PIPELINE: OpenSearch search pipeline that combines BM25 and
      vector search scores using Reciprocal Rank Fusion (RRF). The formula
      score = sum(1/(k+rank)) with k=60 produces balanced rankings without
      requiring manual weight tuning.

How it helps:
    - text_analyzer: "running" matches "run", "runs", "runner" (snowball stemming)
    - strict mapping: prevents indexing bugs (e.g., sending "updated_date" when
      only "updated_at" is mapped — caught immediately, not silently ignored)
    - knn_vector + HNSW: sub-millisecond approximate nearest neighbor search
      even with millions of vectors (ef_construction=512 for high recall)
    - RRF pipeline: combines keyword and semantic results without the fragile
      score normalization that weighted averaging requires
"""

# Simple papers index (Week 3 - BM25 only)
ARXIV_PAPERS_INDEX = "arxiv-papers"

ARXIV_PAPERS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "standard_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_"
                },
                "text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"]
                }
            }
        }
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "arxiv_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 512}}
            },
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}
            },
            "abstract": {
                "type": "text",
                "analyzer": "text_analyzer"
            },
            "categories": {"type": "keyword"},
            "published_date": {"type": "date"},
            "updated_date": {"type": "date"},
            "pdf_url": {"type": "keyword"},
            "pdf_content": {
                "type": "text",
                "analyzer": "text_analyzer"
            },
            "parsing_status": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"}
        }
    }
}


# Chunks index (Week 4-5 - Hybrid search with embeddings)
ARXIV_PAPERS_CHUNKS_INDEX = "arxiv-papers-chunks"

# Index mapping for chunked papers with vector embeddings
ARXIV_PAPERS_CHUNKS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
        "index.knn.space_type": "cosinesimil",
        "analysis": {
            "analyzer": {
                "standard_analyzer": {"type": "standard", "stopwords": "_english_"},
                "text_analyzer": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase", "stop", "snowball"]},
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "keyword"},
            "arxiv_id": {"type": "keyword"},
            "paper_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "chunk_word_count": {"type": "integer"},
            "start_char": {"type": "integer"},
            "end_char": {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,  # Jina v3 embeddings dimension
                "method": {
                    "name": "hnsw",  # Hierarchical Navigable Small World
                    "space_type": "cosinesimil",  # Cosine similarity
                    "engine": "nmslib",
                    "parameters": {
                        "ef_construction": 512,  # Higher value = better recall, slower indexing
                        "m": 16,  # Number of bi-directional links
                    },
                },
            },
            "title": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "authors": {
                "type": "text",
                "analyzer": "standard_analyzer",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "abstract": {"type": "text", "analyzer": "text_analyzer"},
            "categories": {"type": "keyword"},
            "published_date": {"type": "date"},
            "section_title": {"type": "keyword"},
            "embedding_model": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        },
    },
}

HYBRID_RRF_PIPELINE = {
    "id": "hybrid-rrf-pipeline",
    "description": "Post processor for hybrid RRF search",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf",  # Reciprocal Rank Fusion
                    "rank_constant": 60,  # Default k=60 for RRF formula: 1/(k+rank)
                }
            }
        }
    ],
}

# Alternative: Weighted average pipeline (commented out - not used by default)
# This could be used if you need explicit control over BM25 vs vector weights
# However, RRF generally provides better results without manual weight tuning
"""
HYBRID_SEARCH_PIPELINE = {
    "id": "hybrid-ranking-pipeline",
    "description": "Hybrid search pipeline using weighted average for BM25 and vector similarity",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {
                    "technique": "l2"  # L2 normalization for better score distribution
                },
                "combination": {
                    "technique": "harmonic_mean",  # Harmonic mean often works better than arithmetic
                    "parameters": {
                        "weights": [0.3, 0.7]  # 30% BM25, 70% vector similarity
                    }
                }
            }
        }
    ]
}
"""