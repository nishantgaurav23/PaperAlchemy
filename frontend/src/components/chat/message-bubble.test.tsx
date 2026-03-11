import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { MessageBubble } from "./message-bubble";
import type { ChatMessage } from "@/types/chat";

const userMessage: ChatMessage = {
  id: "user-1",
  role: "user",
  content: "What are transformers?",
  timestamp: Date.now(),
};

const assistantMessage: ChatMessage = {
  id: "asst-1",
  role: "assistant",
  content: "Transformers are a neural network architecture [1]. They use self-attention [2].",
  sources: [
    {
      title: "Attention Is All You Need",
      authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
      year: 2017,
      arxiv_id: "1706.03762",
    },
    {
      title: "BERT: Pre-training of Deep Bidirectional Transformers",
      authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
      year: 2018,
      arxiv_id: "1810.04805",
    },
  ],
  timestamp: Date.now(),
};

const errorMessage: ChatMessage = {
  id: "err-1",
  role: "error",
  content: "Network error",
  timestamp: Date.now(),
};

describe("MessageBubble", () => {
  it("renders user message content", () => {
    render(<MessageBubble message={userMessage} />);
    expect(screen.getByText("What are transformers?")).toBeInTheDocument();
  });

  it("renders user message right-aligned", () => {
    render(<MessageBubble message={userMessage} />);
    const container = screen.getByTestId("message-user-1");
    expect(container.className).toContain("justify-end");
  });

  it("renders assistant message left-aligned", () => {
    render(<MessageBubble message={assistantMessage} />);
    const container = screen.getByTestId("message-asst-1");
    expect(container.className).toContain("justify-start");
  });

  it("renders inline citation badges", () => {
    render(<MessageBubble message={assistantMessage} />);
    const badge1 = screen.getByRole("button", { name: "Citation 1" });
    const badge2 = screen.getByRole("button", { name: "Citation 2" });
    expect(badge1).toBeInTheDocument();
    expect(badge2).toBeInTheDocument();
  });

  it("renders source cards with arxiv links", () => {
    render(<MessageBubble message={assistantMessage} />);
    expect(screen.getByText("Attention Is All You Need")).toBeInTheDocument();

    const links = screen.getAllByRole("link");
    const arxivLink = links.find((l) =>
      l.getAttribute("href")?.includes("1706.03762"),
    );
    expect(arxivLink).toBeDefined();
    expect(arxivLink).toHaveAttribute("target", "_blank");
    expect(arxivLink).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("renders source authors truncated to 3", () => {
    render(<MessageBubble message={assistantMessage} />);
    // Second source has 4 authors — should truncate
    expect(screen.getByText(/Jacob Devlin.*et al\./)).toBeInTheDocument();
  });

  it("renders error message with distinct styling", () => {
    render(<MessageBubble message={errorMessage} />);
    expect(screen.getByText("Network error")).toBeInTheDocument();
  });

  it("renders retry button for error messages", () => {
    const onRetry = vi.fn();
    render(<MessageBubble message={errorMessage} onRetry={onRetry} />);

    const retryBtn = screen.getByRole("button", { name: /retry/i });
    expect(retryBtn).toBeInTheDocument();

    fireEvent.click(retryBtn);
    expect(onRetry).toHaveBeenCalled();
  });

  it("does not render retry button without onRetry prop", () => {
    render(<MessageBubble message={errorMessage} />);
    expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
  });

  it("renders assistant message without sources", () => {
    const msgNoSources: ChatMessage = {
      id: "asst-no-sources",
      role: "assistant",
      content: "Just a plain response.",
      timestamp: Date.now(),
    };
    render(<MessageBubble message={msgNoSources} />);
    expect(screen.getByText("Just a plain response.")).toBeInTheDocument();
    expect(screen.queryByText("Sources:")).not.toBeInTheDocument();
  });

  it("renders timestamp", () => {
    render(<MessageBubble message={userMessage} />);
    expect(screen.getByText("just now")).toBeInTheDocument();
  });
});
