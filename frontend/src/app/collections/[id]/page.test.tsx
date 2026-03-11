import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import CollectionDetailPage from "./page";
import { createCollection, addPaper } from "@/lib/collections";
import type { Paper } from "@/types/paper";

const mockLocalStorage = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value; },
    removeItem: (key: string) => { delete store[key]; },
    clear: () => { store = {}; },
    get length() { return Object.keys(store).length; },
    key: (index: number) => Object.keys(store)[index] ?? null,
  };
})();

Object.defineProperty(window, "localStorage", { value: mockLocalStorage, writable: true });

let mockParamsId = "non-existent";

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    back: vi.fn(),
  }),
  useParams: () => ({ id: mockParamsId }),
}));

const mockPaper: Paper = {
  id: "p1",
  arxiv_id: "2301.00001",
  title: "Attention Is All You Need",
  authors: ["Vaswani"],
  abstract: "We propose a new architecture.",
  categories: ["cs.CL"],
  published_date: "2017-06-12",
};

const mockPaper2: Paper = {
  id: "p2",
  arxiv_id: "2301.00002",
  title: "BERT",
  authors: ["Devlin"],
  abstract: "Pre-training deep bidirectional transformers.",
  categories: ["cs.CL"],
  published_date: "2018-10-11",
};

beforeEach(() => {
  localStorage.clear();
  mockParamsId = "non-existent";
});

describe("CollectionDetailPage", () => {
  it("shows not found message for non-existent collection", () => {
    render(<CollectionDetailPage />);
    expect(screen.getByText(/not found/i)).toBeInTheDocument();
  });

  it("renders collection name as heading", () => {
    const col = createCollection("ML Papers", "Great papers");
    mockParamsId = col.id;
    render(<CollectionDetailPage />);
    expect(
      screen.getByRole("heading", { name: "ML Papers" })
    ).toBeInTheDocument();
  });

  it("renders paper count", () => {
    const col = createCollection("Test");
    addPaper(col.id, mockPaper);
    addPaper(col.id, mockPaper2);
    mockParamsId = col.id;
    render(<CollectionDetailPage />);
    expect(screen.getByText(/2 papers/)).toBeInTheDocument();
  });

  it("renders papers in the collection", () => {
    const col = createCollection("Test");
    addPaper(col.id, mockPaper);
    mockParamsId = col.id;
    render(<CollectionDetailPage />);
    expect(screen.getByText("Attention Is All You Need")).toBeInTheDocument();
  });

  it("renders empty state for collection with no papers", () => {
    const col = createCollection("Empty");
    mockParamsId = col.id;
    render(<CollectionDetailPage />);
    expect(screen.getByText(/no papers/i)).toBeInTheDocument();
  });

  it("renders back link to collections", () => {
    const col = createCollection("Test");
    mockParamsId = col.id;
    render(<CollectionDetailPage />);
    const backLink = screen.getByRole("link", {
      name: /back to collections/i,
    });
    expect(backLink).toHaveAttribute("href", "/collections");
  });

  it("renders share button", () => {
    const col = createCollection("Test");
    mockParamsId = col.id;
    render(<CollectionDetailPage />);
    expect(
      screen.getByRole("button", { name: /share/i })
    ).toBeInTheDocument();
  });

  it("removes paper when remove button clicked", () => {
    const col = createCollection("Test");
    addPaper(col.id, mockPaper);
    addPaper(col.id, mockPaper2);
    mockParamsId = col.id;
    render(<CollectionDetailPage />);

    const removeButtons = screen.getAllByRole("button", { name: /remove/i });
    fireEvent.click(removeButtons[0]);

    // After removal, only one paper should remain
    expect(
      screen.queryByText("Attention Is All You Need")
    ).not.toBeInTheDocument();
    expect(screen.getByText("BERT")).toBeInTheDocument();
  });
});
