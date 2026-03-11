"use client";

export function TypingIndicator() {
  return (
    <div className="flex justify-start px-4 py-1">
      <div
        data-testid="typing-indicator"
        className="flex items-center gap-1 rounded-2xl bg-muted px-4 py-3"
      >
        <span className="size-2 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:0ms]" />
        <span className="size-2 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:150ms]" />
        <span className="size-2 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:300ms]" />
      </div>
    </div>
  );
}
