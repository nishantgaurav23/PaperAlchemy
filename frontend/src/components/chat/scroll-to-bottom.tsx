"use client";

import { ArrowDown } from "lucide-react";

interface ScrollToBottomProps {
  onClick: () => void;
  visible: boolean;
}

export function ScrollToBottom({ onClick, visible }: ScrollToBottomProps) {
  if (!visible) return null;

  return (
    <button
      onClick={onClick}
      aria-label="Scroll to bottom"
      className="absolute bottom-20 right-4 z-10 inline-flex size-8 items-center justify-center rounded-full border border-border bg-background shadow-md transition-colors hover:bg-accent"
    >
      <ArrowDown className="size-4" />
    </button>
  );
}
