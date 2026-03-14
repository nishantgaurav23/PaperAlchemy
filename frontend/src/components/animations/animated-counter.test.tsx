import { render, screen, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { AnimatedCounter } from "./animated-counter";

// Mock IntersectionObserver
let observerCallback: IntersectionObserverCallback;

const mockObserverInstance = {
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
};

class MockIntersectionObserver {
  constructor(callback: IntersectionObserverCallback) {
    observerCallback = callback;
    Object.assign(this, mockObserverInstance);
  }
  observe = mockObserverInstance.observe;
  unobserve = mockObserverInstance.unobserve;
  disconnect = mockObserverInstance.disconnect;
}

beforeEach(() => {
  mockObserverInstance.observe.mockClear();
  mockObserverInstance.unobserve.mockClear();
  mockObserverInstance.disconnect.mockClear();
  vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("AnimatedCounter", () => {
  it("renders with initial value of 0", () => {
    render(<AnimatedCounter value={1000} />);
    expect(screen.getByTestId("animated-counter").textContent).toBe("0");
  });

  it("formats numbers with locale separators after animation", () => {
    vi.useFakeTimers();
    render(<AnimatedCounter value={1234} />);
    const el = screen.getByTestId("animated-counter");

    // Trigger intersection
    act(() => {
      observerCallback(
        [
          {
            isIntersecting: true,
            target: el,
          } as IntersectionObserverEntry,
        ],
        mockObserverInstance as unknown as IntersectionObserver
      );
    });

    // Fast-forward animation
    act(() => {
      vi.advanceTimersByTime(2000);
    });

    expect(el.textContent).toBe("1,234");
    vi.useRealTimers();
  });

  it("displays 0 without animation when value is 0", () => {
    render(<AnimatedCounter value={0} />);
    expect(screen.getByTestId("animated-counter").textContent).toBe("0");
  });

  it("abbreviates large numbers when abbreviate is true", () => {
    vi.useFakeTimers();
    render(<AnimatedCounter value={1500} abbreviate />);
    const el = screen.getByTestId("animated-counter");

    act(() => {
      observerCallback(
        [
          {
            isIntersecting: true,
            target: el,
          } as IntersectionObserverEntry,
        ],
        mockObserverInstance as unknown as IntersectionObserver
      );
    });

    act(() => {
      vi.advanceTimersByTime(2000);
    });

    expect(el.textContent).toBe("1.5K");
    vi.useRealTimers();
  });

  it("abbreviates millions", () => {
    vi.useFakeTimers();
    render(<AnimatedCounter value={3400000} abbreviate />);
    const el = screen.getByTestId("animated-counter");

    act(() => {
      observerCallback(
        [
          {
            isIntersecting: true,
            target: el,
          } as IntersectionObserverEntry,
        ],
        mockObserverInstance as unknown as IntersectionObserver
      );
    });

    act(() => {
      vi.advanceTimersByTime(2000);
    });

    expect(el.textContent).toBe("3.4M");
    vi.useRealTimers();
  });

  it("renders prefix and suffix", () => {
    render(<AnimatedCounter value={100} prefix="$" suffix="+" />);
    const el = screen.getByTestId("animated-counter");
    // Shows prefix and suffix around the 0
    expect(el.textContent).toContain("$");
    expect(el.textContent).toContain("+");
  });

  it("accepts custom className", () => {
    render(<AnimatedCounter value={100} className="text-3xl" />);
    const el = screen.getByTestId("animated-counter");
    expect(el.className).toContain("text-3xl");
  });
});
