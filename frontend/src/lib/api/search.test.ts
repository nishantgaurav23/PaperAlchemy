import { describe, it, expect, vi, beforeEach } from "vitest";
import { searchPapers } from "./search";

// Mock the api-client
vi.mock("@/lib/api-client", () => ({
  apiClient: {
    get: vi.fn(),
  },
}));

import { apiClient } from "@/lib/api-client";

const mockGet = vi.mocked(apiClient.get);

describe("searchPapers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls GET /api/v1/search with no params for empty query", async () => {
    mockGet.mockResolvedValue({ papers: [], total: 0, page: 1, page_size: 20, total_pages: 0 });

    await searchPapers({});
    expect(mockGet).toHaveBeenCalledWith("/api/v1/search");
  });

  it("passes query param", async () => {
    mockGet.mockResolvedValue({ papers: [], total: 0, page: 1, page_size: 20, total_pages: 0 });

    await searchPapers({ q: "transformers" });
    expect(mockGet).toHaveBeenCalledWith("/api/v1/search?q=transformers");
  });

  it("passes all params", async () => {
    mockGet.mockResolvedValue({ papers: [], total: 0, page: 1, page_size: 20, total_pages: 0 });

    await searchPapers({ q: "attention", category: "cs.AI", sort: "date_desc", page: 2 });
    const url = mockGet.mock.calls[0][0];
    expect(url).toContain("q=attention");
    expect(url).toContain("category=cs.AI");
    expect(url).toContain("sort=date_desc");
    expect(url).toContain("page=2");
  });

  it("returns the API response", async () => {
    const mockResponse = {
      papers: [{ id: "1", title: "Test" }],
      total: 1,
      page: 1,
      page_size: 20,
      total_pages: 1,
    };
    mockGet.mockResolvedValue(mockResponse);

    const result = await searchPapers({ q: "test" });
    expect(result).toEqual(mockResponse);
  });
});
