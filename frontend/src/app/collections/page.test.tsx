import { render, screen } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import CollectionsPage from "./page";
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

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    back: vi.fn(),
  }),
}));

const mockPaper: Paper = {
  id: "p1",
  arxiv_id: "2301.00001",
  title: "Test Paper",
  authors: ["Author"],
  abstract: "Abstract",
  categories: ["cs.AI"],
  published_date: "2023-01-01",
};

beforeEach(() => {
  localStorage.clear();
});

describe("CollectionsPage", () => {
  it("renders page heading", () => {
    render(<CollectionsPage />);
    expect(screen.getByText("Collections")).toBeInTheDocument();
  });

  it("renders empty state when no collections", () => {
    render(<CollectionsPage />);
    expect(screen.getByText(/create your first collection to start/i)).toBeInTheDocument();
  });

  it("renders collection cards when collections exist", () => {
    createCollection("ML Papers", "Machine learning");
    createCollection("NLP Papers", "NLP research");
    render(<CollectionsPage />);
    expect(screen.getByText("ML Papers")).toBeInTheDocument();
    expect(screen.getByText("NLP Papers")).toBeInTheDocument();
  });

  it("renders create button", () => {
    render(<CollectionsPage />);
    expect(
      screen.getByRole("button", { name: /new collection/i })
    ).toBeInTheDocument();
  });

  it("shows paper count on collection cards", () => {
    const col = createCollection("With Papers");
    addPaper(col.id, mockPaper);
    render(<CollectionsPage />);
    expect(screen.getByText(/1 paper$/)).toBeInTheDocument();
  });
});
