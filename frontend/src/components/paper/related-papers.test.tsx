import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RelatedPapers } from "./related-papers";
import type { Paper } from "@/types/paper";

const mockRelated: Paper[] = [
  {
    id: "def-456",
    arxiv_id: "1810.04805",
    title: "BERT: Pre-training of Deep Bidirectional Transformers",
    authors: ["Jacob Devlin", "Ming-Wei Chang"],
    abstract: "We introduce a new language representation model.",
    categories: ["cs.CL"],
    published_date: "2018-10-11",
  },
  {
    id: "ghi-789",
    arxiv_id: "2005.14165",
    title: "Language Models are Few-Shot Learners",
    authors: ["Tom Brown", "Benjamin Mann"],
    abstract: "We show GPT-3 can perform few-shot learning.",
    categories: ["cs.CL", "cs.AI"],
    published_date: "2020-05-28",
  },
];

describe("RelatedPapers", () => {
  it("renders related paper cards", () => {
    render(<RelatedPapers papers={mockRelated} />);
    expect(screen.getByText(/BERT/)).toBeInTheDocument();
    expect(screen.getByText(/Language Models are Few-Shot Learners/)).toBeInTheDocument();
  });

  it("renders links to paper detail pages", () => {
    render(<RelatedPapers papers={mockRelated} />);
    const bertLink = screen.getByRole("link", { name: /BERT/ });
    expect(bertLink).toHaveAttribute("href", "/papers/def-456");
  });

  it("renders author names (truncated if many)", () => {
    render(<RelatedPapers papers={mockRelated} />);
    expect(screen.getByText(/Jacob Devlin/)).toBeInTheDocument();
  });

  it("renders category badges", () => {
    render(<RelatedPapers papers={mockRelated} />);
    // cs.CL appears on both cards
    const badges = screen.getAllByText("cs.CL");
    expect(badges.length).toBeGreaterThanOrEqual(1);
  });

  it("hides section when papers array is empty", () => {
    const { container } = render(<RelatedPapers papers={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("hides section when papers is undefined", () => {
    const { container } = render(<RelatedPapers papers={undefined} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders section heading", () => {
    render(<RelatedPapers papers={mockRelated} />);
    expect(
      screen.getByRole("heading", { name: /related papers/i })
    ).toBeInTheDocument();
  });
});
