import { apiClient } from "@/lib/api-client";
import type { Paper, PaperDetail, RelatedPapersResponse } from "@/types/paper";

export interface ListPapersParams {
  query?: string;
  category?: string;
  limit?: number;
  offset?: number;
}

export async function listPapers(params: ListPapersParams = {}): Promise<Paper[]> {
  const searchParams = new URLSearchParams();
  if (params.query) searchParams.set("query", params.query);
  if (params.category) searchParams.set("category", params.category);
  if (params.limit !== undefined) searchParams.set("limit", String(params.limit));
  if (params.offset !== undefined) searchParams.set("offset", String(params.offset));

  const qs = searchParams.toString();
  const path = `/api/v1/papers${qs ? `?${qs}` : ""}`;
  return apiClient.get<Paper[]>(path);
}

/** UUID pattern: 8-4-4-4-12 hex characters */
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export async function getPaper(id: string): Promise<PaperDetail> {
  // If the id looks like a UUID, query by UUID; otherwise treat it as an arxiv_id.
  const path = UUID_RE.test(id)
    ? `/api/v1/papers/${id}`
    : `/api/v1/papers/by-arxiv/${encodeURIComponent(id)}`;
  return apiClient.get<PaperDetail>(path);
}

export async function getRelatedPapers(_id: string): Promise<RelatedPapersResponse> {
  // Related papers endpoint not yet implemented — return empty list.
  return { papers: [] };
}

/** Trigger AI analysis generation for a paper and return updated paper detail. */
export async function requestAnalysis(paperId: string): Promise<PaperDetail> {
  // Fire all three analysis requests in parallel (they persist to DB)
  const summaryReq = apiClient.post(`/api/v1/analysis/papers/${paperId}/summary`, {}).catch(() => null);
  const highlightsReq = apiClient.post(`/api/v1/analysis/papers/${paperId}/highlights`, {}).catch(() => null);
  const methodologyReq = apiClient.post(`/api/v1/analysis/papers/${paperId}/methodology`, {}).catch(() => null);

  await Promise.all([summaryReq, highlightsReq, methodologyReq]);

  // Re-fetch the paper with analysis fields now populated
  return getPaper(paperId);
}
