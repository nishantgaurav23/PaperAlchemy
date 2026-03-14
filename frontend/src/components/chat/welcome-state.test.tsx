import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { WelcomeState } from "./welcome-state";

describe("WelcomeState", () => {
  it("renders title and description", () => {
    render(<WelcomeState onSelectQuestion={vi.fn()} />);
    expect(
      screen.getByText("PaperAlchemy Research Assistant"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Ask me anything about research papers/),
    ).toBeInTheDocument();
  });

  it("renders suggested questions", () => {
    render(<WelcomeState onSelectQuestion={vi.fn()} />);
    const buttons = screen.getAllByTestId("suggested-question");
    expect(buttons).toHaveLength(4);
  });

  it("calls onSelectQuestion when clicking a suggestion", () => {
    const onSelect = vi.fn();
    render(<WelcomeState onSelectQuestion={onSelect} />);

    const buttons = screen.getAllByTestId("suggested-question");
    fireEvent.click(buttons[0]);

    expect(onSelect).toHaveBeenCalledWith(
      "What are the latest advances in transformer architectures?",
    );
  });

  it("renders 'Try asking:' label", () => {
    render(<WelcomeState onSelectQuestion={vi.fn()} />);
    expect(screen.getByText("Try asking:")).toBeInTheDocument();
  });

  it("renders gradient icon container", () => {
    render(<WelcomeState onSelectQuestion={vi.fn()} />);
    const iconContainer = screen.getByTestId("welcome-icon");
    expect(iconContainer).toBeInTheDocument();
    expect(iconContainer.className).toContain("bg-gradient-to-br");
  });

  it("renders AI-Powered Research badge", () => {
    render(<WelcomeState onSelectQuestion={vi.fn()} />);
    expect(screen.getByText("AI-Powered Research")).toBeInTheDocument();
  });
});
