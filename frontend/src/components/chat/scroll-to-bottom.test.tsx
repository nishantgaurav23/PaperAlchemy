import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { ScrollToBottom } from "./scroll-to-bottom";

describe("ScrollToBottom", () => {
  it("renders when visible", () => {
    render(<ScrollToBottom visible onClick={vi.fn()} />);
    expect(screen.getByRole("button", { name: /scroll to bottom/i })).toBeInTheDocument();
  });

  it("does not render when not visible", () => {
    render(<ScrollToBottom visible={false} onClick={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /scroll to bottom/i })).not.toBeInTheDocument();
  });

  it("calls onClick when clicked", () => {
    const onClick = vi.fn();
    render(<ScrollToBottom visible onClick={onClick} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalled();
  });
});
