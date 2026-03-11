import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { CategoryFilter } from "./category-filter";

describe("CategoryFilter", () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders a select element with 'All Categories' option", () => {
    render(<CategoryFilter value="" onChange={mockOnChange} />);
    const select = screen.getByLabelText("Category");
    expect(select).toBeInTheDocument();
    expect(screen.getByText("All Categories")).toBeInTheDocument();
  });

  it("renders common arXiv categories", () => {
    render(<CategoryFilter value="" onChange={mockOnChange} />);
    expect(screen.getByText("cs.AI")).toBeInTheDocument();
    expect(screen.getByText("cs.CL")).toBeInTheDocument();
    expect(screen.getByText("cs.LG")).toBeInTheDocument();
    expect(screen.getByText("stat.ML")).toBeInTheDocument();
  });

  it("calls onChange when a category is selected", async () => {
    const user = userEvent.setup();
    render(<CategoryFilter value="" onChange={mockOnChange} />);

    const select = screen.getByLabelText("Category");
    await user.selectOptions(select, "cs.AI");

    expect(mockOnChange).toHaveBeenCalledWith("cs.AI");
  });

  it("shows selected category", () => {
    render(<CategoryFilter value="cs.LG" onChange={mockOnChange} />);
    const select = screen.getByLabelText("Category") as HTMLSelectElement;
    expect(select.value).toBe("cs.LG");
  });

  it("calls onChange with empty string when All Categories is selected", async () => {
    const user = userEvent.setup();
    render(<CategoryFilter value="cs.AI" onChange={mockOnChange} />);

    const select = screen.getByLabelText("Category");
    await user.selectOptions(select, "");

    expect(mockOnChange).toHaveBeenCalledWith("");
  });
});
