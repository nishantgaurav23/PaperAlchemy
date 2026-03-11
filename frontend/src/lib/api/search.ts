import { apiClient } from "@/lib/api-client";
import type { SearchParams, SearchResponse } from "@/types/paper";

export async function searchPapers(params: SearchParams): Promise<SearchResponse> {
  const searchParams = new URLSearchParams();

  if (params.q) searchParams.set("q", params.q);
  if (params.category) searchParams.set("category", params.category);
  if (params.sort) searchParams.set("sort", params.sort);
  if (params.page) searchParams.set("page", String(params.page));

  const query = searchParams.toString();
  const path = `/api/v1/search${query ? `?${query}` : ""}`;

  return apiClient.get<SearchResponse>(path);
}
