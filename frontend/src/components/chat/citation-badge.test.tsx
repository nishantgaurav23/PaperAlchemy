import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { CitationBadge } from "./citation-badge";

describe("CitationBadge", () => {
  it("renders citation number", () => {
    render(<CitationBadge number={1} />);
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("has aria label", () => {
    render(<CitationBadge number={3} />);
    expect(screen.getByRole("button", { name: "Citation 3" })).toBeInTheDocument();
  });

  it("calls onClick when clicked", () => {
    const onClick = vi.fn();
    render(<CitationBadge number={1} onClick={onClick} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalled();
  });
});
