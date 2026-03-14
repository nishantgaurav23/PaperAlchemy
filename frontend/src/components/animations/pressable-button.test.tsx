import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { PressableButton } from "./pressable-button";

describe("PressableButton", () => {
  it("renders children", () => {
    render(<PressableButton>Click me</PressableButton>);
    expect(screen.getByText("Click me")).toBeInTheDocument();
  });

  it("applies scale-95 on mouse down", () => {
    render(<PressableButton data-testid="btn">Press</PressableButton>);
    const btn = screen.getByTestId("btn");
    fireEvent.mouseDown(btn);
    expect(btn.style.transform).toBe("scale(0.95)");
  });

  it("removes scale on mouse up", () => {
    render(<PressableButton data-testid="btn">Press</PressableButton>);
    const btn = screen.getByTestId("btn");
    fireEvent.mouseDown(btn);
    fireEvent.mouseUp(btn);
    expect(btn.style.transform).toBe("scale(1)");
  });

  it("removes scale on mouse leave", () => {
    render(<PressableButton data-testid="btn">Press</PressableButton>);
    const btn = screen.getByTestId("btn");
    fireEvent.mouseDown(btn);
    fireEvent.mouseLeave(btn);
    expect(btn.style.transform).toBe("scale(1)");
  });

  it("does not apply scale when disabled", () => {
    render(
      <PressableButton data-testid="btn" disabled>
        Disabled
      </PressableButton>
    );
    const btn = screen.getByTestId("btn");
    fireEvent.mouseDown(btn);
    expect(btn.style.transform).not.toBe("scale(0.95)");
  });

  it("forwards onClick handler", () => {
    const onClick = vi.fn();
    render(
      <PressableButton data-testid="btn" onClick={onClick}>
        Click
      </PressableButton>
    );
    fireEvent.click(screen.getByTestId("btn"));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("has transition style for smooth effect", () => {
    render(<PressableButton data-testid="btn">Press</PressableButton>);
    const btn = screen.getByTestId("btn");
    expect(btn.style.transition).toContain("transform");
  });

  it("accepts custom className", () => {
    render(
      <PressableButton data-testid="btn" className="bg-red-500">
        Styled
      </PressableButton>
    );
    const btn = screen.getByTestId("btn");
    expect(btn.className).toContain("bg-red-500");
  });
});
