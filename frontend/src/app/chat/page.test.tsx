import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import ChatPage from "./page";

// Mock streamChat
const mockStreamChat = vi.fn();
vi.mock("@/lib/api/chat", () => ({
  streamChat: (...args: unknown[]) => mockStreamChat(...args),
}));

// Mock crypto.randomUUID
const mockUUID = vi.fn().mockReturnValue("test-uuid-123");
Object.defineProperty(globalThis, "crypto", {
  value: { randomUUID: mockUUID },
  writable: true,
});

function setupMockStream(options: {
  tokens?: string[];
  sources?: Array<{
    title: string;
    authors: string[];
    year: number;
    arxiv_id: string;
  }>;
  error?: string;
} = {}) {
  const controller = new AbortController();
  mockStreamChat.mockImplementation(
    (_message: string, _sessionId: string, callbacks: {
      onToken: (token: string) => void;
      onSources: (sources: Array<{
        title: string;
        authors: string[];
        year: number;
        arxiv_id: string;
      }>) => void;
      onDone: () => void;
      onError: (error: string) => void;
    }) => {
      if (options.error) {
        setTimeout(() => callbacks.onError(options.error!), 10);
      } else {
        const tokens = options.tokens ?? ["Hello", " world"];
        let i = 0;
        const interval = setInterval(() => {
          if (i < tokens.length) {
            callbacks.onToken(tokens[i]);
            i++;
          } else {
            clearInterval(interval);
            if (options.sources) {
              callbacks.onSources(options.sources);
            }
            callbacks.onDone();
          }
        }, 5);
      }
      return controller;
    },
  );
  return controller;
}

describe("ChatPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUUID
      .mockReturnValueOnce("session-1")
      .mockReturnValueOnce("msg-1")
      .mockReturnValueOnce("msg-2")
      .mockReturnValueOnce("msg-3");
  });

  it("renders chat page with welcome state", () => {
    render(<ChatPage />);
    expect(screen.getByText("Chat")).toBeInTheDocument();
    expect(
      screen.getByText("PaperAlchemy Research Assistant"),
    ).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: /message input/i })).toBeInTheDocument();
  });

  it("renders new chat button", () => {
    render(<ChatPage />);
    expect(screen.getByRole("button", { name: /new chat/i })).toBeInTheDocument();
  });

  it("shows suggested questions in welcome state", () => {
    render(<ChatPage />);
    const suggestions = screen.getAllByTestId("suggested-question");
    expect(suggestions.length).toBe(4);
  });

  it("sends message on suggested question click", async () => {
    setupMockStream({ tokens: ["Response"] });
    render(<ChatPage />);

    const suggestions = screen.getAllByTestId("suggested-question");
    await act(async () => {
      fireEvent.click(suggestions[0]);
    });

    expect(mockStreamChat).toHaveBeenCalled();
  });

  it("sends message and shows user message", async () => {
    setupMockStream({ tokens: ["Response", " text"] });
    render(<ChatPage />);

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Test question");
    await act(async () => {
      await userEvent.keyboard("{Enter}");
    });

    expect(mockStreamChat).toHaveBeenCalled();
    expect(screen.getByText("Test question")).toBeInTheDocument();
  });

  it("shows streaming response", async () => {
    setupMockStream({
      tokens: ["Streaming", " response"],
      sources: [
        {
          title: "Test Paper",
          authors: ["Author A"],
          year: 2024,
          arxiv_id: "2401.00001",
        },
      ],
    });
    render(<ChatPage />);

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hello");
    await act(async () => {
      await userEvent.keyboard("{Enter}");
    });

    // Wait for streaming to complete
    await waitFor(
      () => {
        expect(screen.getByText(/Streaming response/)).toBeInTheDocument();
      },
      { timeout: 2000 },
    );
  });

  it("shows error message with retry", async () => {
    setupMockStream({ error: "Server error" });
    render(<ChatPage />);

    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Fail me");
    await act(async () => {
      await userEvent.keyboard("{Enter}");
    });

    await waitFor(() => {
      expect(screen.getByText("Server error")).toBeInTheDocument();
    });

    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("clears chat on new chat button", async () => {
    setupMockStream({ tokens: ["Hi"] });
    render(<ChatPage />);

    // Send a message first
    const textarea = screen.getByRole("textbox");
    await userEvent.type(textarea, "Hello");
    await act(async () => {
      await userEvent.keyboard("{Enter}");
    });

    // Wait for streaming to complete and message to appear
    await waitFor(() => {
      expect(screen.getByText("Hello")).toBeInTheDocument();
    });

    // Wait for streaming to fully complete (assistant message finalized)
    await waitFor(() => {
      // Send button should be re-enabled after streaming completes
      expect(screen.getByRole("textbox")).not.toBeDisabled();
    });

    // Click new chat
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /new chat/i }));
    });

    // Welcome state should be back
    await waitFor(() => {
      expect(
        screen.getByText("PaperAlchemy Research Assistant"),
      ).toBeInTheDocument();
    });
  });
});
