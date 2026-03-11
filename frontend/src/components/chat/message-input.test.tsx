import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { MessageInput } from "./message-input";

describe("MessageInput", () => {
  it("renders textarea and send button", () => {
    render(<MessageInput onSubmit={vi.fn()} />);
    expect(screen.getByRole("textbox", { name: /message input/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send message/i })).toBeInTheDocument();
  });

  it("sends message on Enter key", async () => {
    const onSubmit = vi.fn();
    render(<MessageInput onSubmit={onSubmit} />);

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hello world");
    await userEvent.keyboard("{Enter}");

    expect(onSubmit).toHaveBeenCalledWith("Hello world");
  });

  it("adds newline on Shift+Enter", async () => {
    const onSubmit = vi.fn();
    render(<MessageInput onSubmit={onSubmit} />);

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Line 1{Shift>}{Enter}{/Shift}Line 2");

    expect(onSubmit).not.toHaveBeenCalled();
    expect(textarea).toHaveValue("Line 1\nLine 2");
  });

  it("disables send button when empty", () => {
    render(<MessageInput onSubmit={vi.fn()} />);
    const sendBtn = screen.getByRole("button", { name: /send message/i });
    expect(sendBtn).toBeDisabled();
  });

  it("disables send button for whitespace-only input", async () => {
    render(<MessageInput onSubmit={vi.fn()} />);
    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "   ");
    const sendBtn = screen.getByRole("button", { name: /send message/i });
    expect(sendBtn).toBeDisabled();
  });

  it("disables input when streaming", () => {
    render(<MessageInput onSubmit={vi.fn()} isStreaming />);
    expect(screen.getByRole("textbox")).toBeDisabled();
  });

  it("shows stop button when streaming", () => {
    const onStop = vi.fn();
    render(<MessageInput onSubmit={vi.fn()} onStop={onStop} isStreaming />);
    const stopBtn = screen.getByRole("button", { name: /stop generating/i });
    expect(stopBtn).toBeInTheDocument();

    fireEvent.click(stopBtn);
    expect(onStop).toHaveBeenCalled();
  });

  it("clears input after submit", async () => {
    const onSubmit = vi.fn();
    render(<MessageInput onSubmit={onSubmit} />);

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hello");
    await userEvent.keyboard("{Enter}");

    expect(textarea).toHaveValue("");
  });

  it("sends message on send button click", async () => {
    const onSubmit = vi.fn();
    render(<MessageInput onSubmit={onSubmit} />);

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Click send");
    const sendBtn = screen.getByRole("button", { name: /send message/i });
    await userEvent.click(sendBtn);

    expect(onSubmit).toHaveBeenCalledWith("Click send");
  });

  it("respects max message length", async () => {
    render(<MessageInput onSubmit={vi.fn()} />);
    const textarea = screen.getByRole("textbox");
    const longText = "a".repeat(2001);
    // fireEvent to bypass userEvent's character-by-character typing
    fireEvent.change(textarea, { target: { value: longText } });
    // Should not exceed 2000
    expect((textarea as HTMLTextAreaElement).value.length).toBeLessThanOrEqual(2000);
  });
});
