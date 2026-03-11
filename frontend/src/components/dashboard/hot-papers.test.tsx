import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { HotPapers } from "./hot-papers";
import type { Paper } from "@/types/paper";

const mockPapers: Paper[] = [
  {
    id: "hot-001",
    arxiv_id: "2501.12345",
    title: "Scaling Laws for Neural Language Models Revisited",
    authors: ["Alice Chen", "Bob Smith", "Carol Davis"],
    abstract: "We revisit scaling laws.",
    categories: ["cs.CL", "cs.AI"],
    published_date: "2025-01-15",
  },
  {
    id: "hot-002",
    arxiv_id: "2502.67890",
    title: "Efficient Transformers: A Survey",
    authors: ["David Lee"],
    abstract: "A survey of efficient transformers.",
    categories: ["cs.LG"],
    published_date: "2025-02-20",
  },
];

describe("HotPapers", () => {
  it("renders paper titles", () => {
    render(<HotPapers papers={mockPapers} />);

    expect(screen.getByText("Scaling Laws for Neural Language Models Revisited")).toBeInTheDocument();
    expect(screen.getByText("Efficient Transformers: A Survey")).toBeInTheDocument();
  });

  it("renders arXiv links", () => {
    render(<HotPapers papers={mockPapers} />);

    const arxivLinks = screen.getAllByRole("link", { name: /arxiv link/i });
    expect(arxivLinks).toHaveLength(2);
    expect(arxivLinks[0]).toHaveAttribute("href", "https://arxiv.org/abs/2501.12345");
    expect(arxivLinks[0]).toHaveAttribute("target", "_blank");
  });

  it("renders internal paper links", () => {
    render(<HotPapers papers={mockPapers} />);

    const titleLink = screen.getByRole("link", { name: "Scaling Laws for Neural Language Models Revisited" });
    expect(titleLink).toHaveAttribute("href", "/papers/hot-001");
  });

  it("renders authors with et al. for 3+", () => {
    render(<HotPapers papers={mockPapers} />);

    expect(screen.getByText("Alice Chen et al.")).toBeInTheDocument();
    expect(screen.getByText("David Lee")).toBeInTheDocument();
  });

  it("renders category badges", () => {
    render(<HotPapers papers={mockPapers} />);

    expect(screen.getByText("cs.CL")).toBeInTheDocument();
    expect(screen.getByText("cs.AI")).toBeInTheDocument();
    expect(screen.getByText("cs.LG")).toBeInTheDocument();
  });

  it("renders formatted dates", () => {
    render(<HotPapers papers={mockPapers} />);

    expect(screen.getByText("Jan 15, 2025")).toBeInTheDocument();
    expect(screen.getByText("Feb 20, 2025")).toBeInTheDocument();
  });

  it("shows skeleton when loading", () => {
    render(<HotPapers papers={[]} loading />);

    expect(screen.getByTestId("hot-papers-skeleton")).toBeInTheDocument();
  });

  it("shows empty state when no papers", () => {
    render(<HotPapers papers={[]} />);

    expect(screen.getByTestId("hot-papers-empty")).toBeInTheDocument();
    expect(screen.getByText("No papers found")).toBeInTheDocument();
  });

  it("has hot-papers test id", () => {
    render(<HotPapers papers={mockPapers} />);
    expect(screen.getByTestId("hot-papers")).toBeInTheDocument();
  });
});
