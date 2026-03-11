"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Plus } from "lucide-react";
import { MessageBubble } from "@/components/chat/message-bubble";
import { MessageInput } from "@/components/chat/message-input";
import { WelcomeState } from "@/components/chat/welcome-state";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { ScrollToBottom } from "@/components/chat/scroll-to-bottom";
import { streamChat } from "@/lib/api/chat";
import type { ChatMessage, ChatSource } from "@/types/chat";

function generateId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState(() => generateId());
  const [isStreaming, setIsStreaming] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [streamingSources, setStreamingSources] = useState<ChatSource[]>([]);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [streamingTimestamp, setStreamingTimestamp] = useState(0);

  const chatContainerRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const lastUserMessageRef = useRef<string>("");

  const scrollToBottom = useCallback(() => {
    const container = chatContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, []);

  // Auto-scroll when messages change or streaming content updates
  useEffect(() => {
    if (!showScrollButton) {
      scrollToBottom();
    }
  }, [messages, streamingContent, isTyping, showScrollButton, scrollToBottom]);

  const handleScroll = useCallback(() => {
    const container = chatContainerRef.current;
    if (!container) return;
    const threshold = 100;
    const isNearBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
    setShowScrollButton(!isNearBottom);
  }, []);

  const sendMessage = useCallback(
    (content: string) => {
      const userMessage: ChatMessage = {
        id: generateId(),
        role: "user",
        content,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, userMessage]);
      lastUserMessageRef.current = content;
      setIsStreaming(true);
      setIsTyping(true);
      setStreamingContent("");
      setStreamingSources([]);
      setStreamingTimestamp(Date.now());

      const controller = streamChat(content, sessionId, {
        onToken: (token) => {
          setIsTyping(false);
          setStreamingContent((prev) => prev + token);
        },
        onSources: (sources) => {
          setStreamingSources(sources);
        },
        onDone: () => {
          setStreamingContent((prev) => {
            setStreamingSources((sources) => {
              const assistantMessage: ChatMessage = {
                id: generateId(),
                role: "assistant",
                content: prev,
                sources: sources.length > 0 ? sources : undefined,
                timestamp: Date.now(),
              };
              setMessages((msgs) => [...msgs, assistantMessage]);
              return [];
            });
            return "";
          });
          setIsStreaming(false);
          setIsTyping(false);
        },
        onError: (error) => {
          setStreamingContent((prev) => {
            if (prev) {
              // Partial response — save what we have
              const partialMessage: ChatMessage = {
                id: generateId(),
                role: "assistant",
                content: prev,
                timestamp: Date.now(),
              };
              setMessages((msgs) => [...msgs, partialMessage]);
            }
            return "";
          });
          const errorMessage: ChatMessage = {
            id: generateId(),
            role: "error",
            content: error,
            timestamp: Date.now(),
          };
          setMessages((msgs) => [...msgs, errorMessage]);
          setIsStreaming(false);
          setIsTyping(false);
        },
      });

      abortControllerRef.current = controller;
    },
    [sessionId],
  );

  const handleStop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    setStreamingContent((prev) => {
      if (prev) {
        const partialMessage: ChatMessage = {
          id: generateId(),
          role: "assistant",
          content: prev + "\n\n*(Response stopped)*",
          timestamp: Date.now(),
        };
        setMessages((msgs) => [...msgs, partialMessage]);
      }
      return "";
    });

    setIsStreaming(false);
    setIsTyping(false);
  }, []);

  const handleNewChat = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setMessages([]);
    setSessionId(generateId());
    setIsStreaming(false);
    setIsTyping(false);
    setStreamingContent("");
    setStreamingSources([]);
  }, []);

  const handleRetry = useCallback(() => {
    // Remove the last error message and re-send
    setMessages((prev) => {
      const filtered = prev.filter((m) => m.role !== "error");
      return filtered;
    });
    if (lastUserMessageRef.current) {
      sendMessage(lastUserMessageRef.current);
    }
  }, [sendMessage]);

  const handleSelectQuestion = useCallback(
    (question: string) => {
      sendMessage(question);
    },
    [sendMessage],
  );

  const hasMessages = messages.length > 0 || isStreaming;

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <h1 className="text-lg font-semibold">Chat</h1>
        <button
          onClick={handleNewChat}
          aria-label="New chat"
          className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm font-medium transition-colors hover:bg-accent"
        >
          <Plus className="size-4" />
          New Chat
        </button>
      </div>

      {/* Messages area */}
      <div
        ref={chatContainerRef}
        onScroll={handleScroll}
        className="relative flex-1 overflow-y-auto"
      >
        {!hasMessages ? (
          <WelcomeState onSelectQuestion={handleSelectQuestion} />
        ) : (
          <div className="flex flex-col gap-1 py-4">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onRetry={msg.role === "error" ? handleRetry : undefined}
              />
            ))}

            {isTyping && <TypingIndicator />}

            {streamingContent && !isTyping && (
              <MessageBubble
                message={{
                  id: "streaming",
                  role: "assistant",
                  content: streamingContent,
                  sources: streamingSources.length > 0 ? streamingSources : undefined,
                  timestamp: streamingTimestamp,
                }}
              />
            )}
          </div>
        )}

        <ScrollToBottom
          visible={showScrollButton}
          onClick={scrollToBottom}
        />
      </div>

      {/* Input */}
      <MessageInput
        onSubmit={sendMessage}
        onStop={handleStop}
        isStreaming={isStreaming}
      />
    </div>
  );
}
