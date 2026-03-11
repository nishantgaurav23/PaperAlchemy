import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { TypingIndicator } from "./typing-indicator";

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
});
