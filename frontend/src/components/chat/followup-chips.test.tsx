import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { FollowUpChips } from "./followup-chips";

describe("FollowUpChips", () => {
  const suggestions = [
    "Tell me more about attention mechanisms",
    "How does BERT compare?",
    "What are the limitations?",
  ];

  it("renders suggestion chips", () => {
    render(<FollowUpChips suggestions={suggestions} onSelect={vi.fn()} />);
    const chips = screen.getAllByTestId("followup-chip");
    expect(chips).toHaveLength(3);
  });

  it("renders correct suggestion text", () => {
    render(<FollowUpChips suggestions={suggestions} onSelect={vi.fn()} />);
    expect(screen.getByText("Tell me more about attention mechanisms")).toBeInTheDocument();
    expect(screen.getByText("How does BERT compare?")).toBeInTheDocument();
  });

  it("calls onSelect with suggestion text when clicked", () => {
    const onSelect = vi.fn();
    render(<FollowUpChips suggestions={suggestions} onSelect={onSelect} />);

    fireEvent.click(screen.getByText("How does BERT compare?"));
    expect(onSelect).toHaveBeenCalledWith("How does BERT compare?");
  });

  it("renders nothing when suggestions array is empty", () => {
    const { container } = render(<FollowUpChips suggestions={[]} onSelect={vi.fn()} />);
    expect(container.innerHTML).toBe("");
  });

  it("has followup-chips container testid", () => {
    render(<FollowUpChips suggestions={suggestions} onSelect={vi.fn()} />);
    expect(screen.getByTestId("followup-chips")).toBeInTheDocument();
  });
});
