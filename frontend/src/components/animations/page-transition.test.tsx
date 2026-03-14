import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { PageTransition } from "./page-transition";

// Mock framer-motion
vi.mock("framer-motion", () => ({
  motion: {
    div: ({
      children,
      initial,
      animate,
      exit,
      transition,
      ...rest
    }: {
      children?: React.ReactNode;
      initial?: Record<string, unknown>;
      animate?: Record<string, unknown>;
      exit?: Record<string, unknown>;
      transition?: Record<string, unknown>;
      [key: string]: unknown;
    }) => (
      <div
        data-testid="motion-div"
        data-initial={JSON.stringify(initial)}
        data-animate={JSON.stringify(animate)}
        data-exit={JSON.stringify(exit)}
        data-transition={JSON.stringify(transition)}
        {...rest}
      >
        {children}
      </div>
    ),
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="animate-presence">{children}</div>
  ),
}));

describe("PageTransition", () => {
  it("renders children", () => {
    render(
      <PageTransition>
        <p>Page content</p>
      </PageTransition>
    );
    expect(screen.getByText("Page content")).toBeInTheDocument();
  });

  it("applies fade-in with Y-axis slide animation props", () => {
    render(
      <PageTransition>
        <p>Content</p>
      </PageTransition>
    );
    const motionDiv = screen.getByTestId("motion-div");
    const initial = JSON.parse(motionDiv.getAttribute("data-initial") || "{}");
    const animate = JSON.parse(motionDiv.getAttribute("data-animate") || "{}");

    expect(initial.opacity).toBe(0);
    expect(initial.y).toBe(8);
    expect(animate.opacity).toBe(1);
    expect(animate.y).toBe(0);
  });

  it("applies exit animation props", () => {
    render(
      <PageTransition>
        <p>Content</p>
      </PageTransition>
    );
    const motionDiv = screen.getByTestId("motion-div");
    const exit = JSON.parse(motionDiv.getAttribute("data-exit") || "{}");

    expect(exit.opacity).toBe(0);
    expect(exit.y).toBe(-8);
  });

  it("uses ease transition with short duration", () => {
    render(
      <PageTransition>
        <p>Content</p>
      </PageTransition>
    );
    const motionDiv = screen.getByTestId("motion-div");
    const transition = JSON.parse(
      motionDiv.getAttribute("data-transition") || "{}"
    );

    expect(transition.duration).toBeLessThanOrEqual(0.3);
    expect(transition.ease).toBeDefined();
  });

  it("accepts custom className", () => {
    render(
      <PageTransition className="custom-class">
        <p>Content</p>
      </PageTransition>
    );
    const motionDiv = screen.getByTestId("motion-div");
    expect(motionDiv.className).toContain("custom-class");
  });
});
