import { describe, it, expect, vi, beforeEach } from "vitest";
import { searchPapers } from "./search";

// Mock the api-client
vi.mock("@/lib/api-client", () => ({
  apiClient: {
    post: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api-client";

const mockPost = vi.mocked(apiClient.post);

const emptyResponse = {
  query: "",
  total: 0,
  hits: [],
  size: 20,
  from: 0,
  search_mode: "hybrid",
};

describe("searchPapers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls POST /api/v1/search with default body for empty query", async () => {
    mockPost.mockResolvedValue(emptyResponse);

    await searchPapers({});
    expect(mockPost).toHaveBeenCalledWith("/api/v1/search", {
      query: "",
      size: 20,
      from: 0,
      use_hybrid: true,
    });
  });

  it("passes query in POST body", async () => {
    mockPost.mockResolvedValue(emptyResponse);

    await searchPapers({ q: "transformers" });
    expect(mockPost).toHaveBeenCalledWith("/api/v1/search", expect.objectContaining({
      query: "transformers",
    }));
  });

  it("maps category to categories array", async () => {
    mockPost.mockResolvedValue(emptyResponse);

    await searchPapers({ q: "attention", category: "cs.AI" });
    const body = mockPost.mock.calls[0][1] as Record<string, unknown>;
    expect(body.categories).toEqual(["cs.AI"]);
  });

  it("maps sort=date_desc to latest_papers=true", async () => {
    mockPost.mockResolvedValue(emptyResponse);

    await searchPapers({ q: "attention", sort: "date_desc" });
    const body = mockPost.mock.calls[0][1] as Record<string, unknown>;
    expect(body.latest_papers).toBe(true);
  });

  it("computes pagination offset from page number", async () => {
    mockPost.mockResolvedValue(emptyResponse);

    await searchPapers({ q: "test", page: 3 });
    const body = mockPost.mock.calls[0][1] as Record<string, unknown>;
    expect(body.from).toBe(40); // (3-1) * 20
    expect(body.size).toBe(20);
  });

  it("returns the API response", async () => {
    const mockResponse = {
      query: "test",
      total: 1,
      hits: [{ arxiv_id: "2301.00001", title: "Test", authors: [], abstract: "", pdf_url: "", score: 0.9, highlights: {}, chunk_text: "", chunk_id: "", section_title: null }],
      size: 20,
      from: 0,
      search_mode: "hybrid",
    };
    mockPost.mockResolvedValue(mockResponse);

    const result = await searchPapers({ q: "test" });
    expect(result).toEqual(mockResponse);
  });
});
