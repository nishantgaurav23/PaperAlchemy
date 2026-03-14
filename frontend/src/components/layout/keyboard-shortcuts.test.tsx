import { render, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useKeyboardShortcuts } from "./use-keyboard-shortcuts";

const mockPush = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({ push: mockPush })),
}));

// Simple test wrapper component
function TestComponent({ onCommandK }: { onCommandK?: () => void }) {
  useKeyboardShortcuts({ onCommandK });
  return <div data-testid="test-wrapper">Test</div>;
}

describe("useKeyboardShortcuts", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("navigates to /search on Cmd+1", () => {
    render(<TestComponent />);
    fireEvent.keyDown(document, { key: "1", metaKey: true });
    expect(mockPush).toHaveBeenCalledWith("/search");
  });

  it("navigates to /chat on Cmd+2", () => {
    render(<TestComponent />);
    fireEvent.keyDown(document, { key: "2", metaKey: true });
    expect(mockPush).toHaveBeenCalledWith("/chat");
  });

  it("navigates to /upload on Cmd+3", () => {
    render(<TestComponent />);
    fireEvent.keyDown(document, { key: "3", metaKey: true });
    expect(mockPush).toHaveBeenCalledWith("/upload");
  });

  it("navigates to /papers on Cmd+4", () => {
    render(<TestComponent />);
    fireEvent.keyDown(document, { key: "4", metaKey: true });
    expect(mockPush).toHaveBeenCalledWith("/papers");
  });

  it("navigates to /collections on Cmd+5", () => {
    render(<TestComponent />);
    fireEvent.keyDown(document, { key: "5", metaKey: true });
    expect(mockPush).toHaveBeenCalledWith("/collections");
  });

  it("navigates to /dashboard on Cmd+6", () => {
    render(<TestComponent />);
    fireEvent.keyDown(document, { key: "6", metaKey: true });
    expect(mockPush).toHaveBeenCalledWith("/dashboard");
  });

  it("works with Ctrl key (Windows/Linux)", () => {
    render(<TestComponent />);
    fireEvent.keyDown(document, { key: "1", ctrlKey: true });
    expect(mockPush).toHaveBeenCalledWith("/search");
  });

  it("does not navigate when typing in an input field", () => {
    const { container } = render(
      <div>
        <TestComponent />
        <input data-testid="text-input" />
      </div>
    );

    const input = container.querySelector("input")!;
    input.focus();

    fireEvent.keyDown(input, { key: "1", metaKey: true });
    expect(mockPush).not.toHaveBeenCalled();
  });

  it("does not navigate when typing in a textarea", () => {
    const { container } = render(
      <div>
        <TestComponent />
        <textarea data-testid="text-area" />
      </div>
    );

    const textarea = container.querySelector("textarea")!;
    textarea.focus();

    fireEvent.keyDown(textarea, { key: "1", metaKey: true });
    expect(mockPush).not.toHaveBeenCalled();
  });

  it("calls onCommandK when Cmd+K is pressed", () => {
    const onCommandK = vi.fn();
    render(<TestComponent onCommandK={onCommandK} />);
    fireEvent.keyDown(document, { key: "k", metaKey: true });
    expect(onCommandK).toHaveBeenCalled();
  });

  it("does not trigger without modifier key", () => {
    render(<TestComponent />);
    fireEvent.keyDown(document, { key: "1" });
    expect(mockPush).not.toHaveBeenCalled();
  });
});
