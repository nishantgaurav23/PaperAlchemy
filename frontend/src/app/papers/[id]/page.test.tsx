import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock next/navigation
const mockBack = vi.fn();
vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "abc-123" }),
  useRouter: () => ({ back: mockBack }),
}));

// Mock API
vi.mock("@/lib/api/papers", () => ({
  getPaper: vi.fn(),
  getRelatedPapers: vi.fn(),
}));

import PaperDetailPage from "./page";
import { getPaper, getRelatedPapers } from "@/lib/api/papers";

const mockGetPaper = vi.mocked(getPaper);
const mockGetRelated = vi.mocked(getRelatedPapers);

const mockPaper = {
  id: "abc-123",
  arxiv_id: "1706.03762",
  title: "Attention Is All You Need",
  authors: ["Ashish Vaswani", "Noam Shazeer"],
  abstract: "The dominant sequence transduction models...",
  categories: ["cs.CL"],
  published_date: "2017-06-12",
  pdf_url: "https://arxiv.org/pdf/1706.03762",
  sections: [
    { title: "Introduction", content: "Self-attention mechanisms..." },
  ],
  summary: {
    objective: "Replace recurrence",
    method: "Self-attention",
    key_findings: "SOTA translation",
    contribution: "Transformer",
    limitations: "Quadratic",
  },
  highlights: {
    novel_contributions: ["Attention mechanism"],
    important_findings: ["BLEU improvement"],
    practical_implications: ["Parallel training"],
  },
  methodology: {
    approach: "Encoder-decoder",
    datasets: ["WMT"],
    baselines: ["ConvS2S"],
    results: "28.4 BLEU",
  },
};

const mockRelated = {
  papers: [
    {
      id: "def-456",
      arxiv_id: "1810.04805",
      title: "BERT",
      authors: ["Jacob Devlin"],
      abstract: "Language representation model.",
      categories: ["cs.CL"],
      published_date: "2018-10-11",
    },
  ],
};

describe("PaperDetailPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetRelated.mockResolvedValue(mockRelated);
  });

  it("shows loading skeleton initially", () => {
    mockGetPaper.mockReturnValue(new Promise(() => {})); // never resolves
    render(<PaperDetailPage />);
    expect(screen.getByTestId("paper-detail-skeleton")).toBeInTheDocument();
  });

  it("renders paper metadata after loading", async () => {
    mockGetPaper.mockResolvedValue(mockPaper);
    render(<PaperDetailPage />);

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Attention Is All You Need" })
      ).toBeInTheDocument();
    });
    expect(screen.getByText(/Ashish Vaswani/)).toBeInTheDocument();
  });

  it("shows error state with retry button on API failure", async () => {
    mockGetPaper.mockRejectedValue(new Error("Network error"));
    render(<PaperDetailPage />);

    await waitFor(() => {
      expect(screen.getByText(/failed to load paper/i)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("retries fetch on retry button click", async () => {
    mockGetPaper.mockRejectedValueOnce(new Error("Network error"));
    mockGetPaper.mockResolvedValueOnce(mockPaper);
    render(<PaperDetailPage />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Attention Is All You Need" })
      ).toBeInTheDocument();
    });
    expect(mockGetPaper).toHaveBeenCalledTimes(2);
  });

  it("shows not-found state for 404 errors", async () => {
    const error = new Error("API Error 404: Not Found");
    (error as Record<string, unknown>).status = 404;
    mockGetPaper.mockRejectedValue(error);
    render(<PaperDetailPage />);

    await waitFor(() => {
      expect(screen.getByText(/paper not found/i)).toBeInTheDocument();
    });
  });

  it("renders back button that calls router.back()", async () => {
    mockGetPaper.mockResolvedValue(mockPaper);
    render(<PaperDetailPage />);

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Attention Is All You Need" })
      ).toBeInTheDocument();
    });

    const backBtn = screen.getByRole("button", { name: /back/i });
    fireEvent.click(backBtn);
    expect(mockBack).toHaveBeenCalled();
  });

  it("renders related papers section", async () => {
    mockGetPaper.mockResolvedValue(mockPaper);
    render(<PaperDetailPage />);

    await waitFor(() => {
      expect(screen.getByText("BERT")).toBeInTheDocument();
    });
  });

  it("renders paper sections when available", async () => {
    mockGetPaper.mockResolvedValue(mockPaper);
    render(<PaperDetailPage />);

    await waitFor(() => {
      expect(screen.getByText("Introduction")).toBeInTheDocument();
    });
  });

  it("renders analysis tabs when available", async () => {
    mockGetPaper.mockResolvedValue(mockPaper);
    render(<PaperDetailPage />);

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: /summary/i })).toBeInTheDocument();
    });
  });
});
