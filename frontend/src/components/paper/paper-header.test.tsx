import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { PaperHeader } from "./paper-header";
import type { PaperDetail } from "@/types/paper";

const mockPaper: PaperDetail = {
  id: "abc-123",
  arxiv_id: "1706.03762",
  title: "Attention Is All You Need",
  authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
  abstract:
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.",
  categories: ["cs.CL", "cs.AI"],
  published_date: "2017-06-12",
  pdf_url: "https://arxiv.org/pdf/1706.03762",
};

describe("PaperHeader", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders paper title as heading", () => {
    render(<PaperHeader paper={mockPaper} />);
    expect(
      screen.getByRole("heading", { name: "Attention Is All You Need" })
    ).toBeInTheDocument();
  });

  it("renders all authors (not truncated)", () => {
    render(<PaperHeader paper={mockPaper} />);
    expect(screen.getByText(/Ashish Vaswani/)).toBeInTheDocument();
    expect(screen.getByText(/Noam Shazeer/)).toBeInTheDocument();
    expect(screen.getByText(/Niki Parmar/)).toBeInTheDocument();
  });

  it("renders formatted published date", () => {
    render(<PaperHeader paper={mockPaper} />);
    expect(screen.getByText("June 12, 2017")).toBeInTheDocument();
  });

  it("renders category badges", () => {
    render(<PaperHeader paper={mockPaper} />);
    expect(screen.getByText("cs.CL")).toBeInTheDocument();
    expect(screen.getByText("cs.AI")).toBeInTheDocument();
  });

  it("renders full abstract", () => {
    render(<PaperHeader paper={mockPaper} />);
    expect(
      screen.getByText(/The dominant sequence transduction models/)
    ).toBeInTheDocument();
  });

  it("renders arXiv link with target=_blank", () => {
    render(<PaperHeader paper={mockPaper} />);
    const arxivLink = screen.getByRole("link", { name: /arxiv/i });
    expect(arxivLink).toHaveAttribute(
      "href",
      "https://arxiv.org/abs/1706.03762"
    );
    expect(arxivLink).toHaveAttribute("target", "_blank");
  });

  it("renders PDF link with target=_blank", () => {
    render(<PaperHeader paper={mockPaper} />);
    const pdfLink = screen.getByRole("link", { name: /pdf/i });
    expect(pdfLink).toHaveAttribute(
      "href",
      "https://arxiv.org/pdf/1706.03762"
    );
    expect(pdfLink).toHaveAttribute("target", "_blank");
  });

  it("hides PDF link when pdf_url is not provided", () => {
    const paperNoPdf = { ...mockPaper, pdf_url: undefined };
    render(<PaperHeader paper={paperNoPdf} />);
    expect(screen.queryByRole("link", { name: /pdf/i })).not.toBeInTheDocument();
  });

  it("copy citation button writes to clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    render(<PaperHeader paper={mockPaper} />);
    const copyBtn = screen.getByRole("button", { name: /copy citation/i });
    fireEvent.click(copyBtn);

    expect(writeText).toHaveBeenCalledWith(
      expect.stringContaining("Attention Is All You Need")
    );
    expect(writeText).toHaveBeenCalledWith(
      expect.stringContaining("Vaswani")
    );
  });

  it("handles empty authors gracefully", () => {
    const paperNoAuthors = { ...mockPaper, authors: [] };
    render(<PaperHeader paper={paperNoAuthors} />);
    // Should not crash
    expect(
      screen.getByRole("heading", { name: "Attention Is All You Need" })
    ).toBeInTheDocument();
  });

  it("handles empty categories gracefully", () => {
    const paperNoCats = { ...mockPaper, categories: [] };
    render(<PaperHeader paper={paperNoCats} />);
    expect(screen.queryByText("cs.CL")).not.toBeInTheDocument();
  });
});
