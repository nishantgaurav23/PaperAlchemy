import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { SearchBar } from "./search-bar";

describe("SearchBar", () => {
  const mockOnSearch = vi.fn();
  const mockOnClear = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
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
});
