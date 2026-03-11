import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { CollectionCard } from "./collection-card";
import type { Collection } from "@/types/collection";

const mockCollection: Collection = {
  id: "col-1",
  name: "ML Papers",
  description: "Machine learning reading list",
  papers: [
    {
      id: "p1",
      arxiv_id: "2301.00001",
      title: "Paper 1",
      authors: ["Author"],
      abstract: "Abstract",
      categories: ["cs.AI"],
      published_date: "2023-01-01",
    },
    {
      id: "p2",
      arxiv_id: "2301.00002",
      title: "Paper 2",
      authors: ["Author"],
      abstract: "Abstract",
      categories: ["cs.AI"],
      published_date: "2023-01-02",
    },
  ],
  createdAt: "2024-01-01T00:00:00Z",
  updatedAt: "2024-06-15T12:00:00Z",
};

const emptyCollection: Collection = {
  id: "col-2",
  name: "Empty List",
  description: "",
  papers: [],
  createdAt: "2024-01-01T00:00:00Z",
  updatedAt: "2024-01-01T00:00:00Z",
};

describe("CollectionCard", () => {
  it("renders collection name", () => {
    render(<CollectionCard collection={mockCollection} onDelete={vi.fn()} />);
    expect(screen.getByText("ML Papers")).toBeInTheDocument();
  });

  it("renders description preview", () => {
    render(<CollectionCard collection={mockCollection} onDelete={vi.fn()} />);
    expect(
      screen.getByText("Machine learning reading list")
    ).toBeInTheDocument();
  });

  it("renders paper count", () => {
    render(<CollectionCard collection={mockCollection} onDelete={vi.fn()} />);
    expect(screen.getByText(/2 papers/)).toBeInTheDocument();
  });

  it("renders singular 'paper' for count of 1", () => {
    const col = { ...mockCollection, papers: [mockCollection.papers[0]] };
    render(<CollectionCard collection={col} onDelete={vi.fn()} />);
    expect(screen.getByText(/1 paper$/)).toBeInTheDocument();
  });

  it("renders '0 papers' for empty collection", () => {
    render(<CollectionCard collection={emptyCollection} onDelete={vi.fn()} />);
    expect(screen.getByText(/0 papers/)).toBeInTheDocument();
  });

  it("links to collection detail page", () => {
    render(<CollectionCard collection={mockCollection} onDelete={vi.fn()} />);
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "/collections/col-1");
  });

  it("calls onDelete when delete button clicked", () => {
    const onDelete = vi.fn();
    render(<CollectionCard collection={mockCollection} onDelete={onDelete} />);
    const deleteBtn = screen.getByRole("button", { name: /delete/i });
    fireEvent.click(deleteBtn);
    expect(onDelete).toHaveBeenCalledWith("col-1");
  });
});
