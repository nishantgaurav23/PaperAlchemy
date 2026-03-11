import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { PaperList } from "./paper-list";
import type { Paper } from "@/types/paper";

const papers: Paper[] = [
  {
    id: "p1",
    arxiv_id: "2301.00001",
    title: "Paper One",
    authors: ["Author A"],
    abstract: "First abstract",
    categories: ["cs.AI"],
    published_date: "2023-01-01",
  },
  {
    id: "p2",
    arxiv_id: "2301.00002",
    title: "Paper Two",
    authors: ["Author B"],
    abstract: "Second abstract",
    categories: ["cs.CL"],
    published_date: "2023-02-01",
  },
];

describe("PaperList", () => {
  it("renders all papers", () => {
    render(
      <PaperList papers={papers} onRemove={vi.fn()} onReorder={vi.fn()} />
    );
    expect(screen.getByText("Paper One")).toBeInTheDocument();
    expect(screen.getByText("Paper Two")).toBeInTheDocument();
  });

  it("renders remove button for each paper", () => {
    render(
      <PaperList papers={papers} onRemove={vi.fn()} onReorder={vi.fn()} />
    );
    const removeButtons = screen.getAllByRole("button", { name: /remove/i });
    expect(removeButtons).toHaveLength(2);
  });

  it("calls onRemove with paper id when remove clicked", () => {
    const onRemove = vi.fn();
    render(
      <PaperList papers={papers} onRemove={onRemove} onReorder={vi.fn()} />
    );
    const removeButtons = screen.getAllByRole("button", { name: /remove/i });
    fireEvent.click(removeButtons[0]);
    expect(onRemove).toHaveBeenCalledWith("p1");
  });

  it("renders empty state when no papers", () => {
    render(
      <PaperList papers={[]} onRemove={vi.fn()} onReorder={vi.fn()} />
    );
    expect(screen.getByText(/no papers/i)).toBeInTheDocument();
  });

  it("renders arXiv links for papers", () => {
    render(
      <PaperList papers={papers} onRemove={vi.fn()} onReorder={vi.fn()} />
    );
    const links = screen.getAllByRole("link", { name: /arxiv/i });
    expect(links).toHaveLength(2);
    expect(links[0]).toHaveAttribute(
      "href",
      "https://arxiv.org/abs/2301.00001"
    );
  });

  it("renders authors for each paper", () => {
    render(
      <PaperList papers={papers} onRemove={vi.fn()} onReorder={vi.fn()} />
    );
    expect(screen.getByText("Author A")).toBeInTheDocument();
    expect(screen.getByText("Author B")).toBeInTheDocument();
  });

  it("renders drag handles when more than one paper", () => {
    render(
      <PaperList papers={papers} onRemove={vi.fn()} onReorder={vi.fn()} />
    );
    const handles = screen.getAllByLabelText(/drag/i);
    expect(handles).toHaveLength(2);
  });

  it("does not render drag handles for single paper", () => {
    render(
      <PaperList
        papers={[papers[0]]}
        onRemove={vi.fn()}
        onReorder={vi.fn()}
      />
    );
    expect(screen.queryByLabelText(/drag/i)).not.toBeInTheDocument();
  });
});
