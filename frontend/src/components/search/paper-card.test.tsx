import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { PaperCard } from "./paper-card";
import type { Paper } from "@/types/paper";

const mockPaper: Paper = {
  id: "abc-123",
  arxiv_id: "2301.00001",
  title: "Attention Is All You Need",
  authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
  abstract:
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
  categories: ["cs.CL", "cs.AI"],
  published_date: "2017-06-12",
};

const paperManyAuthors: Paper = {
  ...mockPaper,
  id: "def-456",
  authors: [
    "Author One",
    "Author Two",
    "Author Three",
    "Author Four",
    "Author Five",
  ],
};

const paperLongAbstract: Paper = {
  ...mockPaper,
  id: "ghi-789",
  abstract: "A".repeat(300),
};

const paperMissingFields: Paper = {
  id: "jkl-000",
  arxiv_id: "2301.99999",
  title: "Paper With Minimal Data",
  authors: [],
  abstract: "",
  categories: [],
  published_date: "2023-01-01",
};

describe("PaperCard", () => {
  it("renders title, authors, and abstract", () => {
    render(<PaperCard paper={mockPaper} />);

    expect(screen.getByText("Attention Is All You Need")).toBeInTheDocument();
    expect(screen.getByText(/Ashish Vaswani/)).toBeInTheDocument();
    expect(
      screen.getByText(/The dominant sequence transduction/)
    ).toBeInTheDocument();
  });

  it("renders published date formatted", () => {
    render(<PaperCard paper={mockPaper} />);
    expect(screen.getByText("Jun 12, 2017")).toBeInTheDocument();
  });

  it("renders category badges", () => {
    render(<PaperCard paper={mockPaper} />);
    expect(screen.getByText("cs.CL")).toBeInTheDocument();
    expect(screen.getByText("cs.AI")).toBeInTheDocument();
  });

  it("renders arXiv link with target=_blank", () => {
    render(<PaperCard paper={mockPaper} />);
    const arxivLink = screen.getByRole("link", { name: /arxiv/i });
    expect(arxivLink).toHaveAttribute(
      "href",
      "https://arxiv.org/abs/2301.00001"
    );
    expect(arxivLink).toHaveAttribute("target", "_blank");
    expect(arxivLink).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("renders title as link to internal paper detail page", () => {
    render(<PaperCard paper={mockPaper} />);
    const titleLink = screen.getByRole("link", {
      name: "Attention Is All You Need",
    });
    expect(titleLink).toHaveAttribute("href", "/papers/abc-123");
  });

  it("truncates authors to first 3 with et al.", () => {
    render(<PaperCard paper={paperManyAuthors} />);
    expect(screen.getByText(/Author One/)).toBeInTheDocument();
    expect(screen.getByText(/Author Two/)).toBeInTheDocument();
    expect(screen.getByText(/Author Three/)).toBeInTheDocument();
    expect(screen.getByText(/et al\./)).toBeInTheDocument();
    expect(screen.queryByText(/Author Four/)).not.toBeInTheDocument();
  });

  it("truncates long abstracts to 200 chars", () => {
    render(<PaperCard paper={paperLongAbstract} />);
    const abstractEl = screen.getByTestId("paper-abstract");
    // Should show 200 chars + "..."
    expect(abstractEl.textContent!.length).toBeLessThanOrEqual(204);
    expect(abstractEl.textContent).toContain("...");
  });

  it("shows 'Unknown' for missing authors", () => {
    render(<PaperCard paper={paperMissingFields} />);
    expect(screen.getByText("Unknown")).toBeInTheDocument();
  });

  it("handles empty abstract", () => {
    render(<PaperCard paper={paperMissingFields} />);
    // Should not crash, no abstract text shown
    expect(
      screen.queryByTestId("paper-abstract")
    ).not.toBeInTheDocument();
  });

  it("handles empty categories", () => {
    render(<PaperCard paper={paperMissingFields} />);
    // Should not crash, no badges shown
    expect(screen.queryByText("cs.CL")).not.toBeInTheDocument();
  });
});
