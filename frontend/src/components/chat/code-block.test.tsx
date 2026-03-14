import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { CodeBlock } from "./code-block";

describe("CodeBlock", () => {
  beforeEach(() => {
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
  });

  it("renders code content", () => {
    render(<CodeBlock language="python">{"print('hello')"}</CodeBlock>);
    expect(screen.getByText("print('hello')")).toBeInTheDocument();
  });

  it("displays language label", () => {
    render(<CodeBlock language="python">code</CodeBlock>);
    expect(screen.getByText("python")).toBeInTheDocument();
  });

  it("displays 'text' when no language specified", () => {
    render(<CodeBlock>some code</CodeBlock>);
    expect(screen.getByText("text")).toBeInTheDocument();
  });

  it("renders copy button with 'Copy' text", () => {
    render(<CodeBlock language="js">const x = 1;</CodeBlock>);
    expect(screen.getByRole("button", { name: /copy code/i })).toBeInTheDocument();
    expect(screen.getByText("Copy")).toBeInTheDocument();
  });

  it("copies code to clipboard on click", async () => {
    render(<CodeBlock language="js">const x = 1;</CodeBlock>);

    const copyBtn = screen.getByRole("button", { name: /copy code/i });
    fireEvent.click(copyBtn);

    await waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith("const x = 1;");
    });
  });

  it("shows 'Copied!' feedback after clicking copy", async () => {
    render(<CodeBlock language="js">const x = 1;</CodeBlock>);

    fireEvent.click(screen.getByRole("button", { name: /copy code/i }));

    await waitFor(() => {
      expect(screen.getByText("Copied!")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /copied/i })).toBeInTheDocument();
    });
  });

  it("applies language class to code element", () => {
    const { container } = render(<CodeBlock language="python">code</CodeBlock>);
    const codeEl = container.querySelector("code");
    expect(codeEl?.className).toContain("language-python");
  });
});
