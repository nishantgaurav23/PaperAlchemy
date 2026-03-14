import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { SearchResults } from "./search-results";
import type { Paper } from "@/types/paper";

const mockPapers: Paper[] = [
  {
    id: "1",
    arxiv_id: "2301.00001",
    title: "Paper One",
    authors: ["Author A"],
    abstract: "Abstract for paper one.",
    categories: ["cs.AI"],
    published_date: "2023-01-01",
  },
  {
    id: "2",
    arxiv_id: "2301.00002",
    title: "Paper Two",
    authors: ["Author B"],
    abstract: "Abstract for paper two.",
    categories: ["cs.CL"],
    published_date: "2023-02-01",
  },
];

describe("SearchResults", () => {
  it("renders loading skeletons when loading", () => {
    render(
      <SearchResults
        papers={[]}
        total={0}
        page={1}
        totalPages={0}
        pageSize={20}
        isLoading={true}
        error={null}
        onPageChange={vi.fn()}
      />
    );
    const skeletons = screen.getAllByTestId("paper-skeleton");
    expect(skeletons.length).toBeGreaterThanOrEqual(3);
  });

  it("renders error state with retry button", async () => {
    const mockRetry = vi.fn();
    render(
      <SearchResults
        papers={[]}
        total={0}
        page={1}
        totalPages={0}
        pageSize={20}
        isLoading={false}
        error="Failed to fetch"
        onPageChange={vi.fn()}
        onRetry={mockRetry}
      />
    );

    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    const retryButton = screen.getByRole("button", { name: /retry/i });
    expect(retryButton).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(retryButton);
    expect(mockRetry).toHaveBeenCalled();
  });

  it("renders empty state when no results and not loading", () => {
    render(
      <SearchResults
        papers={[]}
        total={0}
        page={1}
        totalPages={0}
        pageSize={20}
        isLoading={false}
        error={null}
        hasSearched={true}
        onPageChange={vi.fn()}
      />
    );
    expect(screen.getByText(/no papers found/i)).toBeInTheDocument();
  });

  it("renders initial state when no search performed", () => {
    render(
      <SearchResults
        papers={[]}
        total={0}
        page={1}
        totalPages={0}
        pageSize={20}
        isLoading={false}
        error={null}
        hasSearched={false}
        onPageChange={vi.fn()}
      />
    );
    expect(
      screen.getByText(/search for papers/i)
    ).toBeInTheDocument();
  });

  it("renders paper cards when results are available", () => {
    render(
      <SearchResults
        papers={mockPapers}
        total={2}
        page={1}
        totalPages={1}
        pageSize={20}
        isLoading={false}
        error={null}
        hasSearched={true}
        onPageChange={vi.fn()}
      />
    );
    expect(screen.getByText("Paper One")).toBeInTheDocument();
    expect(screen.getByText("Paper Two")).toBeInTheDocument();
  });

  it("renders pagination when multiple pages", () => {
    render(
      <SearchResults
        papers={mockPapers}
        total={50}
        page={1}
        totalPages={3}
        pageSize={20}
        isLoading={false}
        error={null}
        hasSearched={true}
        onPageChange={vi.fn()}
      />
    );
    expect(screen.getByLabelText("Next page")).toBeInTheDocument();
  });

  // --- New S13.4 tests ---

  it("renders shimmer skeletons during loading", () => {
    render(
      <SearchResults
        papers={[]}
        total={0}
        page={1}
        totalPages={0}
        pageSize={20}
        isLoading={true}
        error={null}
        onPageChange={vi.fn()}
      />
    );
    const skeletons = screen.getAllByTestId("paper-skeleton");
    expect(skeletons.length).toBe(4);
    // Skeletons should use glass-card styling
    expect(skeletons[0].className).toContain("glass-card");
  });

  it("renders enhanced empty state with search query", () => {
    render(
      <SearchResults
        papers={[]}
        total={0}
        page={1}
        totalPages={0}
        pageSize={20}
        isLoading={false}
        error={null}
        hasSearched={true}
        searchQuery="nonexistent topic"
        onPageChange={vi.fn()}
      />
    );
    expect(screen.getByText(/nonexistent topic/)).toBeInTheDocument();
  });

  it("renders search suggestions in empty state when onSuggestionClick provided", () => {
    render(
      <SearchResults
        papers={[]}
        total={0}
        page={1}
        totalPages={0}
        pageSize={20}
        isLoading={false}
        error={null}
        hasSearched={true}
        onPageChange={vi.fn()}
        onSuggestionClick={vi.fn()}
      />
    );
    expect(screen.getByText("transformer architecture")).toBeInTheDocument();
    expect(screen.getByText("large language models")).toBeInTheDocument();
  });

  it("calls onSuggestionClick when a suggestion is clicked", async () => {
    const mockSuggestionClick = vi.fn();
    const user = userEvent.setup();
    render(
      <SearchResults
        papers={[]}
        total={0}
        page={1}
        totalPages={0}
        pageSize={20}
        isLoading={false}
        error={null}
        hasSearched={true}
        onPageChange={vi.fn()}
        onSuggestionClick={mockSuggestionClick}
      />
    );

    await user.click(screen.getByText("transformer architecture"));
    expect(mockSuggestionClick).toHaveBeenCalledWith("transformer architecture");
  });

  it("truncates long search query in empty state message", () => {
    const longQuery = "a".repeat(100);
    render(
      <SearchResults
        papers={[]}
        total={0}
        page={1}
        totalPages={0}
        pageSize={20}
        isLoading={false}
        error={null}
        hasSearched={true}
        searchQuery={longQuery}
        onPageChange={vi.fn()}
      />
    );
    expect(screen.getByText(/\.\.\./)).toBeInTheDocument();
  });
});
