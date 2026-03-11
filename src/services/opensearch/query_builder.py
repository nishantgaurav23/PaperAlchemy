"""OpenSearch query builder for constructing BM25 search queries.

Builds complex OpenSearch queries with proper scoring, filtering,
highlighting, and sorting for both paper-level and chunk-level search.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Unified query builder for OpenSearch BM25 search queries."""

    def __init__(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: list[str] | None = None,
        categories: list[str] | None = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
        search_chunks: bool = True,
    ):
        self.query = query
        self.size = size
        self.from_ = from_
        self.categories = categories
        self.track_total_hits = track_total_hits
        self.latest_papers = latest_papers
        self.search_chunks = search_chunks

        if fields is None:
            if search_chunks:
                self.fields = ["chunk_text^3", "title^2", "abstract^1"]
            else:
                self.fields = ["title^3", "abstract^2", "authors^1"]
        else:
            self.fields = fields

    def build(self) -> dict[str, Any]:
        """Build the complete OpenSearch query body."""
        query_body: dict[str, Any] = {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": self.track_total_hits,
            "_source": self._build_source_fields(),
            "highlight": self._build_highlight(),
        }
        sort = self._build_sort()
        if sort:
            query_body["sort"] = sort
        return query_body

    def _build_query(self) -> dict[str, Any]:
        must_clauses: list[dict[str, Any]] = []
        if self.query.strip():
            must_clauses.append(self._build_text_query())

        bool_query: dict[str, Any] = {}
        bool_query["must"] = must_clauses if must_clauses else [{"match_all": {}}]

        filter_clauses = self._build_filters()
        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {"bool": bool_query}

    def _build_text_query(self) -> dict[str, Any]:
        return {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO",
                "prefix_length": 2,
            }
        }

    def _build_filters(self) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        if self.categories:
            filters.append({"terms": {"categories": self.categories}})
        return filters

    def _build_source_fields(self) -> Any:
        if self.search_chunks:
            return {"excludes": ["embedding"]}
        return [
            "arxiv_id",
            "title",
            "authors",
            "abstract",
            "categories",
            "published_date",
            "pdf_url",
        ]

    def _build_highlight(self) -> dict[str, Any]:
        if self.search_chunks:
            return {
                "fields": {
                    "chunk_text": {
                        "fragment_size": 150,
                        "number_of_fragments": 2,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                    "title": {
                        "fragment_size": 0,
                        "number_of_fragments": 0,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                    "abstract": {
                        "fragment_size": 150,
                        "number_of_fragments": 1,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                },
                "require_field_match": False,
            }
        return {
            "fields": {
                "title": {"fragment_size": 0, "number_of_fragments": 0},
                "abstract": {
                    "fragment_size": 150,
                    "number_of_fragments": 3,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
                "authors": {
                    "fragment_size": 0,
                    "number_of_fragments": 0,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
            },
            "require_field_match": False,
        }

    def _build_sort(self) -> list[dict[str, Any]] | None:
        if self.latest_papers:
            return [{"published_date": {"order": "desc"}}, "_score"]
        if self.query.strip():
            return None
        return [{"published_date": {"order": "desc"}}, "_score"]
