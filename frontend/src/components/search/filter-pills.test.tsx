import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { FilterPills } from "./filter-pills";

describe("FilterPills", () => {
  const onRemoveCategory = vi.fn();
  const onRemoveSort = vi.fn();
  const onClearQuery = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders category pill when category is set", () => {
    render(
      <FilterPills
        query=""
        category="cs.AI"
        sort="relevance"
        onRemoveCategory={onRemoveCategory}
        onRemoveSort={onRemoveSort}
        onClearQuery={onClearQuery}
      />
    );
    expect(screen.getByText(/cs\.AI/)).toBeInTheDocument();
  });

  it("renders query pill when query is set", () => {
    render(
      <FilterPills
        query="transformers"
        category=""
        sort="relevance"
        onRemoveCategory={onRemoveCategory}
        onRemoveSort={onRemoveSort}
        onClearQuery={onClearQuery}
      />
    );
    expect(screen.getByText(/transformers/)).toBeInTheDocument();
  });

  it("does not render sort pill for default 'relevance'", () => {
    render(
      <FilterPills
        query=""
        category=""
        sort="relevance"
        onRemoveCategory={onRemoveCategory}
        onRemoveSort={onRemoveSort}
        onClearQuery={onClearQuery}
      />
    );
    expect(screen.queryByText(/Relevance/)).not.toBeInTheDocument();
  });

  it("renders sort pill for non-default sort", () => {
    render(
      <FilterPills
        query=""
        category=""
        sort="date_desc"
        onRemoveCategory={onRemoveCategory}
        onRemoveSort={onRemoveSort}
        onClearQuery={onClearQuery}
      />
    );
    expect(screen.getByText(/Newest First/)).toBeInTheDocument();
  });

  it("calls onRemoveCategory when category pill is dismissed", async () => {
    const user = userEvent.setup();
    render(
      <FilterPills
        query=""
        category="cs.CL"
        sort="relevance"
        onRemoveCategory={onRemoveCategory}
        onRemoveSort={onRemoveSort}
        onClearQuery={onClearQuery}
      />
    );
    const removeBtn = screen.getByLabelText("Remove category filter");
    await user.click(removeBtn);
    expect(onRemoveCategory).toHaveBeenCalled();
  });

  it("calls onRemoveSort when sort pill is dismissed", async () => {
    const user = userEvent.setup();
    render(
      <FilterPills
        query=""
        category=""
        sort="date_asc"
        onRemoveCategory={onRemoveCategory}
        onRemoveSort={onRemoveSort}
        onClearQuery={onClearQuery}
      />
    );
    const removeBtn = screen.getByLabelText("Remove sort filter");
    await user.click(removeBtn);
    expect(onRemoveSort).toHaveBeenCalled();
  });

  it("calls onClearQuery when query pill is dismissed", async () => {
    const user = userEvent.setup();
    render(
      <FilterPills
        query="bert"
        category=""
        sort="relevance"
        onRemoveCategory={onRemoveCategory}
        onRemoveSort={onRemoveSort}
        onClearQuery={onClearQuery}
      />
    );
    const removeBtn = screen.getByLabelText("Remove query filter");
    await user.click(removeBtn);
    expect(onClearQuery).toHaveBeenCalled();
  });

  it("renders nothing when no active filters", () => {
    const { container } = render(
      <FilterPills
        query=""
        category=""
        sort="relevance"
        onRemoveCategory={onRemoveCategory}
        onRemoveSort={onRemoveSort}
        onClearQuery={onClearQuery}
      />
    );
    expect(container.firstChild).toBeNull();
  });
});
