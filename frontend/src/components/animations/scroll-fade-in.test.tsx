import { render, screen, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ScrollFadeIn } from "./scroll-fade-in";

// Mock IntersectionObserver
let observerCallback: IntersectionObserverCallback;
let observerOptions: IntersectionObserverInit | undefined;

const mockObserverInstance = {
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
};

class MockIntersectionObserver {
  constructor(callback: IntersectionObserverCallback, options?: IntersectionObserverInit) {
    observerCallback = callback;
    observerOptions = options;
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

describe("ScrollFadeIn", () => {
  it("renders children", () => {
    render(
      <ScrollFadeIn>
        <p>Card content</p>
      </ScrollFadeIn>
    );
    expect(screen.getByText("Card content")).toBeInTheDocument();
  });

  it("starts with opacity 0 and translateY", () => {
    render(
      <ScrollFadeIn>
        <div data-testid="child">Content</div>
      </ScrollFadeIn>
    );
    const wrapper = screen.getByTestId("child").parentElement!;
    expect(wrapper.style.opacity).toBe("0");
    expect(wrapper.style.transform).toContain("translateY");
  });

  it("becomes visible when intersecting", () => {
    render(
      <ScrollFadeIn>
        <div data-testid="child">Content</div>
      </ScrollFadeIn>
    );
    const wrapper = screen.getByTestId("child").parentElement!;

    act(() => {
      observerCallback(
        [{ isIntersecting: true, target: wrapper } as IntersectionObserverEntry],
        mockObserverInstance as unknown as IntersectionObserver
      );
    });

    expect(wrapper.style.opacity).toBe("1");
    expect(wrapper.style.transform).toContain("translateY(0");
  });

  it("uses IntersectionObserver with threshold", () => {
    render(
      <ScrollFadeIn>
        <div>Content</div>
      </ScrollFadeIn>
    );
    expect(observerOptions?.threshold).toBeDefined();
  });

  it("unobserves after becoming visible (animate once)", () => {
    render(
      <ScrollFadeIn>
        <div data-testid="child">Content</div>
      </ScrollFadeIn>
    );
    const wrapper = screen.getByTestId("child").parentElement!;

    act(() => {
      observerCallback(
        [{ isIntersecting: true, target: wrapper } as IntersectionObserverEntry],
        mockObserverInstance as unknown as IntersectionObserver
      );
    });

    expect(mockObserverInstance.unobserve).toHaveBeenCalled();
  });

  it("applies stagger delay", () => {
    render(
      <ScrollFadeIn delay={200}>
        <div data-testid="child">Content</div>
      </ScrollFadeIn>
    );
    const wrapper = screen.getByTestId("child").parentElement!;
    expect(wrapper.style.transitionDelay).toBe("200ms");
  });

  it("accepts custom className", () => {
    render(
      <ScrollFadeIn className="my-class">
        <div data-testid="child">Content</div>
      </ScrollFadeIn>
    );
    const wrapper = screen.getByTestId("child").parentElement!;
    expect(wrapper.className).toContain("my-class");
  });
});
