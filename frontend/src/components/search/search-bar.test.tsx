import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { SearchBar } from "./search-bar";

function createStorageMock() {
  const store = new Map<string, string>();
  return {
    getItem: vi.fn((key: string) => store.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => store.set(key, value)),
    removeItem: vi.fn((key: string) => store.delete(key)),
    clear: vi.fn(() => store.clear()),
    get length() { return store.size; },
    key: vi.fn((index: number) => [...store.keys()][index] ?? null),
    _store: store,
  };
}

describe("SearchBar", () => {
  const mockOnSearch = vi.fn();
  const mockOnClear = vi.fn();
  let storageMock: ReturnType<typeof createStorageMock>;

  beforeEach(() => {
    vi.clearAllMocks();
    storageMock = createStorageMock();
    vi.stubGlobal("localStorage", storageMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders search input with placeholder", () => {
    render(<SearchBar value="" onSearch={mockOnSearch} onClear={mockOnClear} />);
    const input = screen.getByPlaceholderText(
      "Search papers by title, author, or topic..."
    );
    expect(input).toBeInTheDocument();
  });

  it("renders search icon", () => {
    render(<SearchBar value="" onSearch={mockOnSearch} onClear={mockOnClear} />);
    expect(screen.getByTestId("search-icon")).toBeInTheDocument();
  });

  it("calls onSearch when form is submitted", async () => {
    const user = userEvent.setup();
    render(<SearchBar value="" onSearch={mockOnSearch} onClear={mockOnClear} />);

    const input = screen.getByPlaceholderText(
      "Search papers by title, author, or topic..."
    );
    await user.type(input, "transformers");
    await user.keyboard("{Enter}");

    expect(mockOnSearch).toHaveBeenCalledWith("transformers");
  });

  it("shows clear button when value is non-empty", () => {
    render(
      <SearchBar value="test query" onSearch={mockOnSearch} onClear={mockOnClear} />
    );
    expect(screen.getByLabelText("Clear search")).toBeInTheDocument();
  });

  it("hides clear button when value is empty", () => {
    render(<SearchBar value="" onSearch={mockOnSearch} onClear={mockOnClear} />);
    expect(screen.queryByLabelText("Clear search")).not.toBeInTheDocument();
  });

  it("calls onClear when clear button is clicked", async () => {
    const user = userEvent.setup();
    render(
      <SearchBar value="test query" onSearch={mockOnSearch} onClear={mockOnClear} />
    );

    await user.click(screen.getByLabelText("Clear search"));
    expect(mockOnClear).toHaveBeenCalled();
  });

  it("displays the provided value", () => {
    render(
      <SearchBar value="existing query" onSearch={mockOnSearch} onClear={mockOnClear} />
    );
    const input = screen.getByPlaceholderText(
      "Search papers by title, author, or topic..."
    );
    expect(input).toHaveValue("existing query");
  });

  // --- New S13.4 tests: Recent searches ---

  it("shows recent searches dropdown on focus when there are recent searches", async () => {
    storageMock._store.set(
      "paperalchemy:recent-searches",
      JSON.stringify(["attention", "bert"])
    );
    const user = userEvent.setup();
    render(<SearchBar value="" onSearch={mockOnSearch} onClear={mockOnClear} />);

    const input = screen.getByPlaceholderText(
      "Search papers by title, author, or topic..."
    );
    await user.click(input);

    expect(screen.getByTestId("recent-searches-dropdown")).toBeInTheDocument();
    expect(screen.getByText("attention")).toBeInTheDocument();
    expect(screen.getByText("bert")).toBeInTheDocument();
  });

  it("does not show dropdown when no recent searches", async () => {
    const user = userEvent.setup();
    render(<SearchBar value="" onSearch={mockOnSearch} onClear={mockOnClear} />);

    const input = screen.getByPlaceholderText(
      "Search papers by title, author, or topic..."
    );
    await user.click(input);

    expect(screen.queryByTestId("recent-searches-dropdown")).not.toBeInTheDocument();
  });

  it("selects a recent search and calls onSearch", async () => {
    storageMock._store.set(
      "paperalchemy:recent-searches",
      JSON.stringify(["transformers"])
    );
    const user = userEvent.setup();
    render(<SearchBar value="" onSearch={mockOnSearch} onClear={mockOnClear} />);

    const input = screen.getByPlaceholderText(
      "Search papers by title, author, or topic..."
    );
    await user.click(input);
    await user.click(screen.getByText("transformers"));

    expect(mockOnSearch).toHaveBeenCalledWith("transformers");
  });

  it("saves a search to recent on submit", async () => {
    const user = userEvent.setup();
    render(<SearchBar value="" onSearch={mockOnSearch} onClear={mockOnClear} />);

    const input = screen.getByPlaceholderText(
      "Search papers by title, author, or topic..."
    );
    await user.type(input, "new query");
    await user.keyboard("{Enter}");

    expect(storageMock.setItem).toHaveBeenCalledWith(
      "paperalchemy:recent-searches",
      expect.stringContaining("new query")
    );
  });

  it("can remove individual recent search", async () => {
    storageMock._store.set(
      "paperalchemy:recent-searches",
      JSON.stringify(["attention", "bert"])
    );
    const user = userEvent.setup();
    render(<SearchBar value="" onSearch={mockOnSearch} onClear={mockOnClear} />);

    const input = screen.getByPlaceholderText(
      "Search papers by title, author, or topic..."
    );
    await user.click(input);

    const removeBtn = screen.getByLabelText(
      'Remove "attention" from recent searches'
    );
    await user.click(removeBtn);

    expect(screen.queryByText("attention")).not.toBeInTheDocument();
    expect(screen.getByText("bert")).toBeInTheDocument();
  });
});
