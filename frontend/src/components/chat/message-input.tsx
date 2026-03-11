"use client";

import { useRef, useState, useCallback, type KeyboardEvent } from "react";
import { Send, Square } from "lucide-react";
import { MAX_MESSAGE_LENGTH } from "@/types/chat";

interface MessageInputProps {
  onSubmit: (message: string) => void;
  onStop?: () => void;
  disabled?: boolean;
  isStreaming?: boolean;
}

export function MessageInput({
  onSubmit,
  onStop,
  disabled = false,
  isStreaming = false,
}: MessageInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const trimmed = value.trim();
  const canSubmit = trimmed.length > 0 && !disabled && !isStreaming;

  const handleSubmit = useCallback(() => {
    if (!canSubmit) return;
    onSubmit(trimmed);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [canSubmit, trimmed, onSubmit]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const handleInput = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      const maxHeight = 5 * 24; // ~5 lines
      textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
    }
  }, []);

  return (
    <div className="flex items-end gap-2 border-t border-border bg-background p-4">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => {
          if (e.target.value.length <= MAX_MESSAGE_LENGTH) {
            setValue(e.target.value);
          }
        }}
        onKeyDown={handleKeyDown}
        onInput={handleInput}
        placeholder="Ask a research question..."
        disabled={disabled || isStreaming}
        rows={1}
        aria-label="Message input"
        className="flex-1 resize-none rounded-lg border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
      />
      {isStreaming ? (
        <button
          onClick={onStop}
          aria-label="Stop generating"
          className="inline-flex size-9 shrink-0 items-center justify-center rounded-lg bg-destructive text-destructive-foreground transition-colors hover:bg-destructive/80"
        >
          <Square className="size-4" />
        </button>
      ) : (
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          aria-label="Send message"
          className="inline-flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-colors hover:bg-primary/80 disabled:pointer-events-none disabled:opacity-50"
        >
          <Send className="size-4" />
        </button>
      )}
    </div>
  );
}
