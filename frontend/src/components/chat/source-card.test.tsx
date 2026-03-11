import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SourceCard } from "./source-card";
import type { ChatSource } from "@/types/chat";

const mockSource: ChatSource = {
  title: "Attention Is All You Need",
  authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
  year: 2017,
  arxiv_id: "1706.03762",
};

const sourceManyAuthors: ChatSource = {
  title: "BERT Paper",
  authors: ["Author A", "Author B", "Author C", "Author D"],
  year: 2018,
  arxiv_id: "1810.04805",
};

describe("SourceCard", () => {
  it("renders paper title", () => {
    render(<SourceCard source={mockSource} index={0} />);
    expect(screen.getByText("Attention Is All You Need")).toBeInTheDocument();
  });

  it("renders arxiv link with correct href", () => {
    render(<SourceCard source={mockSource} index={0} />);
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "https://arxiv.org/abs/1706.03762");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("renders authors and year", () => {
    render(<SourceCard source={mockSource} index={0} />);
    expect(
      screen.getByText(/Ashish Vaswani, Noam Shazeer, Niki Parmar \(2017\)/),
    ).toBeInTheDocument();
  });

  it("truncates authors to first 3 with et al.", () => {
    render(<SourceCard source={sourceManyAuthors} index={0} />);
    expect(screen.getByText(/Author A.*et al\.\s*\(2018\)/)).toBeInTheDocument();
  });

  it("renders 1-indexed number badge", () => {
    render(<SourceCard source={mockSource} index={0} />);
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("has correct data-testid", () => {
    render(<SourceCard source={mockSource} index={0} />);
    expect(screen.getByTestId("source-card-1")).toBeInTheDocument();
  });
});
