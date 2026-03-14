import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { HoverCardPreview } from "./hover-card-preview";

describe("HoverCardPreview", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  const paper = {
    title: "Attention Is All You Need",
    authors: ["Vaswani", "Shazeer", "Parmar"],
    year: 2017,
    abstract:
      "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
  };

  it("renders the trigger element", () => {
    render(
      <HoverCardPreview paper={paper}>
        <a href="#">Paper Link</a>
      </HoverCardPreview>
    );
    expect(screen.getByText("Paper Link")).toBeInTheDocument();
  });

  it("shows preview card on mouse enter", () => {
    render(
      <HoverCardPreview paper={paper}>
        <span data-testid="trigger">Paper Link</span>
      </HoverCardPreview>
    );
    fireEvent.mouseEnter(screen.getByTestId("trigger"));
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.getByText("Attention Is All You Need")).toBeInTheDocument();
    expect(screen.getByText(/2017/)).toBeInTheDocument();
  });

  it("displays authors in the preview", () => {
    render(
      <HoverCardPreview paper={paper}>
        <span data-testid="trigger">Link</span>
      </HoverCardPreview>
    );
    fireEvent.mouseEnter(screen.getByTestId("trigger"));
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.getByText(/Vaswani/)).toBeInTheDocument();
  });

  it("truncates abstract to 150 characters", () => {
    const longAbstract = "A".repeat(200);
    render(
      <HoverCardPreview paper={{ ...paper, abstract: longAbstract }}>
        <span data-testid="trigger">Link</span>
      </HoverCardPreview>
    );
    fireEvent.mouseEnter(screen.getByTestId("trigger"));
    act(() => {
      vi.advanceTimersByTime(100);
    });
    const abstractEl = screen.getByTestId("hover-abstract");
    expect(abstractEl.textContent!.length).toBeLessThanOrEqual(153); // 150 + "..."
  });

  it("hides preview card on mouse leave with delay", () => {
    render(
      <HoverCardPreview paper={paper}>
        <span data-testid="trigger">Link</span>
      </HoverCardPreview>
    );

    fireEvent.mouseEnter(screen.getByTestId("trigger"));
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.getByText("Attention Is All You Need")).toBeInTheDocument();

    fireEvent.mouseLeave(screen.getByTestId("trigger"));
    // Should still be visible during delay
    expect(screen.getByText("Attention Is All You Need")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(
      screen.queryByText("Attention Is All You Need")
    ).not.toBeInTheDocument();
  });

  it("does not show preview if abstract is missing", () => {
    render(
      <HoverCardPreview paper={{ ...paper, abstract: undefined }}>
        <span data-testid="trigger">Link</span>
      </HoverCardPreview>
    );
    fireEvent.mouseEnter(screen.getByTestId("trigger"));
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.queryByTestId("hover-abstract")).not.toBeInTheDocument();
  });
});
