import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { ToastProvider } from "./toast-provider";

vi.mock("sonner", () => ({
  Toaster: ({
    position,
    visibleToasts,
    richColors,
    ...rest
  }: {
    position?: string;
    visibleToasts?: number;
    richColors?: boolean;
    [key: string]: unknown;
  }) => (
    <div
      data-testid="sonner-toaster"
      data-position={position}
      data-visible-toasts={visibleToasts}
      data-rich-colors={richColors ? "true" : "false"}
      {...rest}
    />
  ),
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
  },
}));

describe("ToastProvider", () => {
  it("renders the sonner Toaster component", () => {
    render(<ToastProvider />);
    expect(screen.getByTestId("sonner-toaster")).toBeInTheDocument();
  });

  it("positions toasts at bottom-right", () => {
    render(<ToastProvider />);
    const toaster = screen.getByTestId("sonner-toaster");
    expect(toaster.getAttribute("data-position")).toBe("bottom-right");
  });

  it("limits visible toasts to 3", () => {
    render(<ToastProvider />);
    const toaster = screen.getByTestId("sonner-toaster");
    expect(toaster.getAttribute("data-visible-toasts")).toBe("3");
  });

  it("enables rich colors", () => {
    render(<ToastProvider />);
    const toaster = screen.getByTestId("sonner-toaster");
    expect(toaster.getAttribute("data-rich-colors")).toBe("true");
  });
});
