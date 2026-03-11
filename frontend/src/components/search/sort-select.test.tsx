import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { SortSelect } from "./sort-select";

describe("SortSelect", () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders a select element with sort options", () => {
    render(<SortSelect value="relevance" onChange={mockOnChange} />);
    const select = screen.getByLabelText("Sort by");
    expect(select).toBeInTheDocument();
    expect(screen.getByText("Relevance")).toBeInTheDocument();
    expect(screen.getByText("Newest First")).toBeInTheDocument();
    expect(screen.getByText("Oldest First")).toBeInTheDocument();
  });

  it("shows selected sort option", () => {
    render(<SortSelect value="date_desc" onChange={mockOnChange} />);
    const select = screen.getByLabelText("Sort by") as HTMLSelectElement;
    expect(select.value).toBe("date_desc");
  });

  it("calls onChange when sort is changed", async () => {
    const user = userEvent.setup();
    render(<SortSelect value="relevance" onChange={mockOnChange} />);

    const select = screen.getByLabelText("Sort by");
    await user.selectOptions(select, "date_desc");

    expect(mockOnChange).toHaveBeenCalledWith("date_desc");
  });
});
