import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { Pagination } from "./pagination";

describe("Pagination", () => {
  const mockOnPageChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders page info with result range", () => {
    render(
      <Pagination
        currentPage={1}
        totalPages={8}
        totalResults={142}
        pageSize={20}
        onPageChange={mockOnPageChange}
      />
    );
    expect(screen.getByText(/Showing 1–20 of 142/)).toBeInTheDocument();
  });

  it("renders correct range for middle page", () => {
    render(
      <Pagination
        currentPage={3}
        totalPages={8}
        totalResults={142}
        pageSize={20}
        onPageChange={mockOnPageChange}
      />
    );
    expect(screen.getByText(/Showing 41–60 of 142/)).toBeInTheDocument();
  });

  it("renders correct range for last page", () => {
    render(
      <Pagination
        currentPage={8}
        totalPages={8}
        totalResults={142}
        pageSize={20}
        onPageChange={mockOnPageChange}
      />
    );
    expect(screen.getByText(/Showing 141–142 of 142/)).toBeInTheDocument();
  });

  it("disables previous button on first page", () => {
    render(
      <Pagination
        currentPage={1}
        totalPages={5}
        totalResults={100}
        pageSize={20}
        onPageChange={mockOnPageChange}
      />
    );
    expect(screen.getByLabelText("Previous page")).toBeDisabled();
  });

  it("disables next button on last page", () => {
    render(
      <Pagination
        currentPage={5}
        totalPages={5}
        totalResults={100}
        pageSize={20}
        onPageChange={mockOnPageChange}
      />
    );
    expect(screen.getByLabelText("Next page")).toBeDisabled();
  });

  it("calls onPageChange when next is clicked", async () => {
    const user = userEvent.setup();
    render(
      <Pagination
        currentPage={1}
        totalPages={5}
        totalResults={100}
        pageSize={20}
        onPageChange={mockOnPageChange}
      />
    );

    await user.click(screen.getByLabelText("Next page"));
    expect(mockOnPageChange).toHaveBeenCalledWith(2);
  });

  it("calls onPageChange when previous is clicked", async () => {
    const user = userEvent.setup();
    render(
      <Pagination
        currentPage={3}
        totalPages={5}
        totalResults={100}
        pageSize={20}
        onPageChange={mockOnPageChange}
      />
    );

    await user.click(screen.getByLabelText("Previous page"));
    expect(mockOnPageChange).toHaveBeenCalledWith(2);
  });

  it("calls onPageChange when a page number is clicked", async () => {
    const user = userEvent.setup();
    render(
      <Pagination
        currentPage={1}
        totalPages={5}
        totalResults={100}
        pageSize={20}
        onPageChange={mockOnPageChange}
      />
    );

    await user.click(screen.getByText("3"));
    expect(mockOnPageChange).toHaveBeenCalledWith(3);
  });

  it("does not render when totalPages is 0", () => {
    const { container } = render(
      <Pagination
        currentPage={1}
        totalPages={0}
        totalResults={0}
        pageSize={20}
        onPageChange={mockOnPageChange}
      />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders ellipsis for large page counts", () => {
    render(
      <Pagination
        currentPage={5}
        totalPages={20}
        totalResults={400}
        pageSize={20}
        onPageChange={mockOnPageChange}
      />
    );
    const ellipses = screen.getAllByText("...");
    expect(ellipses.length).toBeGreaterThan(0);
  });
});
