import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { streamChat } from "./chat";

describe("streamChat", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("calls onToken with streaming tokens", async () => {
    const onToken = vi.fn();
    const onSources = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();

    streamChat("test message", "session-1", {
      onToken,
      onSources,
      onDone,
      onError,
    });

    // Advance through token intervals
    for (let i = 0; i < 100; i++) {
      vi.advanceTimersByTime(20);
    }

    expect(onToken).toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });

  it("calls onDone after all tokens", async () => {
    const onToken = vi.fn();
    const onSources = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();

    streamChat("test", "session-1", {
      onToken,
      onSources,
      onDone,
      onError,
    });

    // Advance enough for all tokens to complete
    for (let i = 0; i < 2000; i++) {
      vi.advanceTimersByTime(20);
    }

    expect(onDone).toHaveBeenCalled();
    expect(onSources).toHaveBeenCalled();
  });

  it("returns an AbortController", () => {
    const controller = streamChat("test", "session-1", {
      onToken: vi.fn(),
      onSources: vi.fn(),
      onDone: vi.fn(),
      onError: vi.fn(),
    });

    expect(controller).toBeInstanceOf(AbortController);
  });

  it("stops streaming on abort", () => {
    const onToken = vi.fn();
    const onDone = vi.fn();

    const controller = streamChat("test", "session-1", {
      onToken,
      onSources: vi.fn(),
      onDone,
      onError: vi.fn(),
    });

    vi.advanceTimersByTime(20);
    const countBefore = onToken.mock.calls.length;

    controller.abort();
    vi.advanceTimersByTime(1000);

    // Should not have received many more tokens after abort
    expect(onToken.mock.calls.length).toBeLessThanOrEqual(countBefore + 1);
  });
});
