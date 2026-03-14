import { apiClient } from "@/lib/api-client";
import type { SearchParams, SearchResponse, ArxivSearchResponse } from "@/types/paper";
import { PAGE_SIZE } from "@/types/paper";

/**
 * Search papers via POST /api/v1/search (hybrid BM25 + KNN).
 *
 * Maps frontend SearchParams to the backend HybridSearchRequest schema.
 */
export async function searchPapers(params: SearchParams): Promise<SearchResponse> {
  const page = params.page ?? 1;
  const size = PAGE_SIZE;
  const from = (page - 1) * size;

  const body: Record<string, unknown> = {
    query: params.q ?? "",
    size,
    from: from,
    use_hybrid: true,
  };

  // Map frontend category filter to backend categories array
  if (params.category) {
    body.categories = [params.category];
  }

  // Map frontend sort option to backend latest_papers flag
  if (params.sort === "date_desc") {
    body.latest_papers = true;
  }

  return apiClient.post<SearchResponse>("/api/v1/search", body);
}

/**
 * Search arXiv directly for papers online via POST /api/v1/search/arxiv.
 *
 * This is a live web search — not limited to the local knowledge base.
 */
export async function searchArxiv(params: {
  query: string;
  category?: string;
  maxResults?: number;
  sortBy?: string;
}): Promise<ArxivSearchResponse> {
  const body: Record<string, unknown> = {
    query: params.query,
    max_results: params.maxResults ?? PAGE_SIZE,
  };

  if (params.category) {
    body.category = params.category;
  }

  if (params.sortBy === "date_desc") {
    body.sort_by = "date";
  }

  return apiClient.post<ArxivSearchResponse>("/api/v1/search/arxiv", body);
}
