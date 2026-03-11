import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { AddToCollection } from "./add-to-collection";
import { createCollection } from "@/lib/collections";
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

const mockPaper: Paper = {
  id: "paper-1",
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

describe("AddToCollection", () => {
  it("renders the trigger button", () => {
    render(<AddToCollection paper={mockPaper} />);
    expect(
      screen.getByRole("button", { name: /add to collection/i })
    ).toBeInTheDocument();
  });

  it("shows collection list when opened", () => {
    createCollection("ML Papers");
    createCollection("NLP Papers");
    render(<AddToCollection paper={mockPaper} />);
    fireEvent.click(
      screen.getByRole("button", { name: /add to collection/i })
    );
    expect(screen.getByText("ML Papers")).toBeInTheDocument();
    expect(screen.getByText("NLP Papers")).toBeInTheDocument();
  });

  it("shows 'Create New' option", () => {
    render(<AddToCollection paper={mockPaper} />);
    fireEvent.click(
      screen.getByRole("button", { name: /add to collection/i })
    );
    expect(screen.getByText(/create new/i)).toBeInTheDocument();
  });

  it("shows empty state when no collections exist", () => {
    render(<AddToCollection paper={mockPaper} />);
    fireEvent.click(
      screen.getByRole("button", { name: /add to collection/i })
    );
    expect(screen.getByText(/no collections/i)).toBeInTheDocument();
  });

  it("calls onAdded callback when paper is added to collection", () => {
    const col = createCollection("Test");
    const onAdded = vi.fn();
    render(<AddToCollection paper={mockPaper} onAdded={onAdded} />);
    fireEvent.click(
      screen.getByRole("button", { name: /add to collection/i })
    );
    fireEvent.click(screen.getByText("Test"));
    expect(onAdded).toHaveBeenCalledWith(col.id);
  });
});
