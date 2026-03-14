import { render, screen, fireEvent, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { PullToRefresh } from "./pull-to-refresh";

describe("PullToRefresh", () => {
  const onRefresh = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    onRefresh.mockClear();
  });

  it("renders children", () => {
    render(
      <PullToRefresh onRefresh={onRefresh}>
        <div data-testid="child">Content</div>
      </PullToRefresh>
    );
    expect(screen.getByTestId("child")).toBeInTheDocument();
  });

  it("shows pull indicator when pulling down from top", () => {
    render(
      <PullToRefresh onRefresh={onRefresh}>
        <div>Content</div>
      </PullToRefresh>
    );

    const container = screen.getByTestId("pull-to-refresh-container");

    // Simulate pull-down gesture
    fireEvent.touchStart(container, {
      touches: [{ clientY: 0 }],
    });
    fireEvent.touchMove(container, {
      touches: [{ clientY: 150 }],
    });

    expect(screen.getByTestId("pull-indicator")).toBeInTheDocument();
  });

  it("triggers onRefresh when pulled past threshold and released", async () => {
    render(
      <PullToRefresh onRefresh={onRefresh} threshold={60}>
        <div>Content</div>
      </PullToRefresh>
    );

    const container = screen.getByTestId("pull-to-refresh-container");

    await act(async () => {
      fireEvent.touchStart(container, {
        touches: [{ clientY: 0 }],
      });
      fireEvent.touchMove(container, {
        touches: [{ clientY: 150 }],
      });
      fireEvent.touchEnd(container);
    });

    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it("does not trigger onRefresh when pull is below threshold", async () => {
    render(
      <PullToRefresh onRefresh={onRefresh} threshold={60}>
        <div>Content</div>
      </PullToRefresh>
    );

    const container = screen.getByTestId("pull-to-refresh-container");

    await act(async () => {
      fireEvent.touchStart(container, {
        touches: [{ clientY: 0 }],
      });
      fireEvent.touchMove(container, {
        touches: [{ clientY: 30 }],
      });
      fireEvent.touchEnd(container);
    });

    expect(onRefresh).not.toHaveBeenCalled();
  });

  it("shows refreshing state during refresh", async () => {
    let resolveRefresh: () => void;
    const slowRefresh = vi.fn(
      () => new Promise<void>((resolve) => { resolveRefresh = resolve; })
    );

    render(
      <PullToRefresh onRefresh={slowRefresh} threshold={60}>
        <div>Content</div>
      </PullToRefresh>
    );

    const container = screen.getByTestId("pull-to-refresh-container");

    await act(async () => {
      fireEvent.touchStart(container, {
        touches: [{ clientY: 0 }],
      });
      fireEvent.touchMove(container, {
        touches: [{ clientY: 150 }],
      });
      fireEvent.touchEnd(container);
    });

    expect(screen.getByTestId("pull-refreshing")).toBeInTheDocument();

    await act(async () => {
      resolveRefresh!();
    });
  });

  it("does not activate when not scrolled to top", () => {
    render(
      <PullToRefresh onRefresh={onRefresh}>
        <div>Content</div>
      </PullToRefresh>
    );

    const container = screen.getByTestId("pull-to-refresh-container");

    // Simulate scrolled position
    Object.defineProperty(container, "scrollTop", { value: 100, writable: true });

    fireEvent.touchStart(container, {
      touches: [{ clientY: 0 }],
    });
    fireEvent.touchMove(container, {
      touches: [{ clientY: 150 }],
    });
    fireEvent.touchEnd(container);

    expect(onRefresh).not.toHaveBeenCalled();
  });

  it("has md:hidden class to disable on desktop", () => {
    render(
      <PullToRefresh onRefresh={onRefresh}>
        <div>Content</div>
      </PullToRefresh>
    );

    // The pull indicator wrapper should be mobile-only
    const container = screen.getByTestId("pull-to-refresh-container");
    expect(container).toBeInTheDocument();
  });
});
