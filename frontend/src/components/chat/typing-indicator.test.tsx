import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TypingIndicator } from "./typing-indicator";

// Mock framer-motion to avoid animation issues in tests
vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>) => {
      const { initial: _i, animate: _a, exit: _e, transition: _t, ...rest } = props;
      return <div data-testid-motion="div" {...rest}>{children}</div>;
    },
    span: ({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>) => {
      const { initial: _i2, animate: _a2, exit: _e2, transition: _t2, ...rest } = props;
      return <span data-testid-motion="span" {...rest}>{children}</span>;
    },
  },
  AnimatePresence: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

describe("TypingIndicator", () => {
  it("renders typing indicator", () => {
    render(<TypingIndicator />);
    expect(screen.getByTestId("typing-indicator")).toBeInTheDocument();
  });

  it("renders three animated dots", () => {
    render(<TypingIndicator />);
    const container = screen.getByTestId("typing-indicator");
    const dots = container.querySelectorAll("span");
    expect(dots).toHaveLength(3);
  });

  it("has framer-motion wrapper for entrance/exit animation", () => {
    const { container } = render(<TypingIndicator />);
    const motionDiv = container.querySelector("[data-testid-motion='div']");
    expect(motionDiv).toBeInTheDocument();
  });
});
