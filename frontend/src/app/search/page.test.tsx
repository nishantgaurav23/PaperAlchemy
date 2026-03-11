import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock next/navigation
const mockPush = vi.fn();
const mockSearchParams = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
  useSearchParams: () => mockSearchParams,
  usePathname: () => "/search",
}));

// Mock the search API
vi.mock("@/lib/api/search", () => ({
  searchPapers: vi.fn(),
}));

import SearchPage from "./page";
import { searchPapers } from "@/lib/api/search";

const mockSearchPapers = vi.mocked(searchPapers);

const mockResponse = {
  papers: [
    {
      id: "1",
      arxiv_id: "2301.00001",
      title: "Test Paper",
      authors: ["John Doe"],
      abstract: "A test abstract.",
      categories: ["cs.AI"],
      published_date: "2023-01-01",
    },
  ],
  total: 1,
  page: 1,
  page_size: 20,
  total_pages: 1,
};

describe("SearchPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSearchPapers.mockResolvedValue(mockResponse);
    // Reset search params
    for (const key of [...mockSearchParams.keys()]) {
      mockSearchParams.delete(key);
    }
  });

  it("renders the search page with input and filters", () => {
    render(<SearchPage />);

    expect(
      screen.getByPlaceholderText("Search papers by title, author, or topic...")
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Category")).toBeInTheDocument();
    expect(screen.getByLabelText("Sort by")).toBeInTheDocument();
  });

  it("renders page title", () => {
    render(<SearchPage />);
    expect(
      screen.getByRole("heading", { name: /search papers/i })
    ).toBeInTheDocument();
  });

  it("shows initial state when no query", () => {
    render(<SearchPage />);
    expect(screen.getByText(/search for papers/i)).toBeInTheDocument();
  });

  it("performs search on form submit", async () => {
    const user = userEvent.setup();
    render(<SearchPage />);

    const input = screen.getByPlaceholderText(
      "Search papers by title, author, or topic..."
    );
    await user.type(input, "transformers");
    await user.keyboard("{Enter}");

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalled();
    });
  });

  it("restores state from URL params", () => {
    mockSearchParams.set("q", "attention");
    mockSearchParams.set("category", "cs.AI");
    mockSearchParams.set("sort", "date_desc");

    render(<SearchPage />);

    const input = screen.getByPlaceholderText(
      "Search papers by title, author, or topic..."
    ) as HTMLInputElement;
    expect(input.value).toBe("attention");
  });

  it("fetches results when query param is present", async () => {
    mockSearchParams.set("q", "neural networks");

    render(<SearchPage />);

    await waitFor(() => {
      expect(mockSearchPapers).toHaveBeenCalledWith(
        expect.objectContaining({ q: "neural networks" })
      );
    });
  });

  it("displays results after search", async () => {
    mockSearchParams.set("q", "test");

    render(<SearchPage />);

    await waitFor(() => {
      expect(screen.getByText("Test Paper")).toBeInTheDocument();
    });
  });

  it("shows error state on API failure", async () => {
    mockSearchPapers.mockRejectedValue(new Error("Network error"));
    mockSearchParams.set("q", "failing query");

    render(<SearchPage />);

    await waitFor(() => {
      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    });
  });
});
