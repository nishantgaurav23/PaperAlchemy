import { describe, it, expect, vi, beforeEach } from "vitest";
import { getPaper, getRelatedPapers, listPapers } from "./papers";

vi.mock("@/lib/api-client", () => ({
  apiClient: {
    get: vi.fn(),
  },
  ApiError: class ApiError extends Error {
    constructor(
      public status: number,
      public statusText: string,
      public body: unknown,
    ) {
      super(`API Error ${status}: ${statusText}`);
      this.name = "ApiError";
    }
  },
}));

import { apiClient, ApiError } from "@/lib/api-client";

const mockGet = vi.mocked(apiClient.get);

const mockPaperDetail = {
  id: "abc-123",
  arxiv_id: "1706.03762",
  title: "Attention Is All You Need",
  authors: ["Ashish Vaswani", "Noam Shazeer"],
  abstract: "The dominant sequence transduction models...",
  categories: ["cs.CL", "cs.AI"],
  published_date: "2017-06-12",
  pdf_url: "https://arxiv.org/pdf/1706.03762",
  sections: [
    { title: "Introduction", content: "Self-attention mechanisms..." },
    { title: "Methods", content: "Multi-head attention..." },
  ],
  summary: {
    objective: "Replace recurrence with attention",
    method: "Multi-head self-attention",
    key_findings: "Achieves SOTA on translation",
    contribution: "Transformer architecture",
    limitations: "Quadratic complexity",
  },
  highlights: {
    novel_contributions: ["Self-attention mechanism"],
    important_findings: ["BLEU score improvement"],
    practical_implications: ["Parallelizable training"],
  },
  methodology: {
    approach: "Encoder-decoder with attention",
    datasets: ["WMT 2014"],
    baselines: ["ConvS2S", "ByteNet"],
    results: "28.4 BLEU on EN-DE",
  },
};

describe("listPapers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls GET /api/v1/papers with no params", async () => {
    mockGet.mockResolvedValue([]);

    await listPapers();
    expect(mockGet).toHaveBeenCalledWith("/api/v1/papers");
  });

  it("calls GET /api/v1/papers with query params", async () => {
    mockGet.mockResolvedValue([]);

    await listPapers({ query: "attention", category: "cs.AI", limit: 10, offset: 0 });
    expect(mockGet).toHaveBeenCalledWith(
      "/api/v1/papers?query=attention&category=cs.AI&limit=10&offset=0",
    );
  });

  it("returns list of papers", async () => {
    const mockPapers = [
      {
        id: "abc-123",
        arxiv_id: "1706.03762",
        title: "Attention Is All You Need",
        authors: ["Ashish Vaswani"],
        abstract: "...",
        categories: ["cs.CL"],
        published_date: "2017-06-12",
      },
    ];
    mockGet.mockResolvedValue(mockPapers);

    const result = await listPapers();
    expect(result).toHaveLength(1);
    expect(result[0].title).toBe("Attention Is All You Need");
  });
});

describe("getPaper", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls GET /api/v1/papers/{id}", async () => {
    mockGet.mockResolvedValue(mockPaperDetail);

    await getPaper("abc-123");
    expect(mockGet).toHaveBeenCalledWith("/api/v1/papers/abc-123");
  });

  it("returns the paper detail response", async () => {
    mockGet.mockResolvedValue(mockPaperDetail);

    const result = await getPaper("abc-123");
    expect(result).toEqual(mockPaperDetail);
  });

  it("throws ApiError on 404", async () => {
    mockGet.mockRejectedValue(new ApiError(404, "Not Found", { detail: "Paper not found" }));

    await expect(getPaper("nonexistent")).rejects.toThrow("API Error 404");
  });
});

describe("getRelatedPapers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls GET /api/v1/papers/{id}/related", async () => {
    mockGet.mockResolvedValue({ papers: [] });

    await getRelatedPapers("abc-123");
    expect(mockGet).toHaveBeenCalledWith("/api/v1/papers/abc-123/related");
  });

  it("returns the related papers list", async () => {
    const mockRelated = {
      papers: [
        {
          id: "def-456",
          arxiv_id: "1810.04805",
          title: "BERT",
          authors: ["Jacob Devlin"],
          categories: ["cs.CL"],
          published_date: "2018-10-11",
        },
      ],
    };
    mockGet.mockResolvedValue(mockRelated);

    const result = await getRelatedPapers("abc-123");
    expect(result.papers).toHaveLength(1);
    expect(result.papers[0].title).toBe("BERT");
  });
});
