import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MobileFilterSheet } from "./mobile-filter-sheet";

describe("MobileFilterSheet", () => {
  const defaultProps = {
    category: "",
    sort: "relevance",
    onCategoryChange: vi.fn(),
    onSortChange: vi.fn(),
  };

  beforeEach(() => {
    defaultProps.onCategoryChange.mockClear();
    defaultProps.onSortChange.mockClear();
  });

  it("renders the filter trigger button", () => {
    render(<MobileFilterSheet {...defaultProps} />);
    expect(screen.getByRole("button", { name: /filters/i })).toBeInTheDocument();
  });

  it("trigger button is hidden on md+ screens", () => {
    render(<MobileFilterSheet {...defaultProps} />);
    const button = screen.getByRole("button", { name: /filters/i });
    expect(button.className).toContain("md:hidden");
  });

  it("opens sheet when trigger button is clicked", () => {
    render(<MobileFilterSheet {...defaultProps} />);
    fireEvent.click(screen.getByRole("button", { name: /filters/i }));
    // Sheet title is rendered inside the sheet content dialog
    expect(screen.getByText("Filter and sort search results")).toBeInTheDocument();
  });

  it("shows category filter controls inside sheet", () => {
    render(<MobileFilterSheet {...defaultProps} />);
    fireEvent.click(screen.getByRole("button", { name: /filters/i }));
    expect(screen.getByText("Category")).toBeInTheDocument();
  });

  it("shows sort controls inside sheet", () => {
    render(<MobileFilterSheet {...defaultProps} />);
    fireEvent.click(screen.getByRole("button", { name: /filters/i }));
    expect(screen.getByText("Sort by")).toBeInTheDocument();
  });

  it("displays active filter count on trigger badge", () => {
    render(<MobileFilterSheet {...defaultProps} category="cs.AI" sort="date_desc" />);
    expect(screen.getByTestId("filter-count")).toHaveTextContent("2");
  });

  it("shows no badge when no filters are active", () => {
    render(<MobileFilterSheet {...defaultProps} />);
    expect(screen.queryByTestId("filter-count")).not.toBeInTheDocument();
  });

  it("calls onCategoryChange when a category is selected", () => {
    render(<MobileFilterSheet {...defaultProps} />);
    fireEvent.click(screen.getByRole("button", { name: /filters/i }));

    const categoryOption = screen.getByRole("button", { name: /cs\.AI/i });
    fireEvent.click(categoryOption);

    expect(defaultProps.onCategoryChange).toHaveBeenCalledWith("cs.AI");
  });

  it("calls onSortChange when a sort option is selected", () => {
    render(<MobileFilterSheet {...defaultProps} />);
    fireEvent.click(screen.getByRole("button", { name: /filters/i }));

    const sortOption = screen.getByRole("button", { name: /newest first/i });
    fireEvent.click(sortOption);

    expect(defaultProps.onSortChange).toHaveBeenCalledWith("date_desc");
  });

  it("has min 44px touch targets for filter options", () => {
    render(<MobileFilterSheet {...defaultProps} />);
    fireEvent.click(screen.getByRole("button", { name: /filters/i }));

    // Check category and sort buttons inside the sheet (not the sheet close button)
    const sheetContent = document.querySelector("[data-slot='sheet-content']");
    expect(sheetContent).not.toBeNull();
    const filterButtons = Array.from(sheetContent!.querySelectorAll("button[aria-label]"));
    expect(filterButtons.length).toBeGreaterThan(0);
    for (const btn of filterButtons) {
      expect(btn.className).toContain("min-h-[44px]");
    }
  });
});
