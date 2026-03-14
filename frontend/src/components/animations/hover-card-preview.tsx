"use client";

import { useState, useRef, useCallback } from "react";
import { cn } from "@/lib/utils";

interface PaperPreview {
  title: string;
  authors: string[];
  year: number;
  abstract?: string;
}

interface HoverCardPreviewProps {
  paper: PaperPreview;
  children: React.ReactNode;
  className?: string;
}

export function HoverCardPreview({
  paper,
  children,
  className,
}: HoverCardPreviewProps) {
  const [visible, setVisible] = useState(false);
  const hideTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = useCallback(() => {
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
      hideTimeoutRef.current = null;
    }
    setVisible(true);
  }, []);

  const hide = useCallback(() => {
    hideTimeoutRef.current = setTimeout(() => {
      setVisible(false);
    }, 200);
  }, []);

  const truncatedAbstract =
    paper.abstract && paper.abstract.length > 150
      ? paper.abstract.slice(0, 150) + "..."
      : paper.abstract;

  return (
    <div className={cn("relative inline-block", className)}>
      <div onMouseEnter={show} onMouseLeave={hide}>
        {children}
      </div>
      {visible && (
        <div
          data-testid="hover-card"
          onMouseEnter={show}
          onMouseLeave={hide}
          className="absolute left-0 top-full z-50 mt-2 w-72 rounded-lg border bg-popover p-4 text-popover-foreground shadow-lg animate-in fade-in-0 zoom-in-95"
        >
          <h4 className="text-sm font-semibold leading-tight">
            {paper.title}
          </h4>
          <p className="mt-1 text-xs text-muted-foreground">
            {paper.authors.join(", ")} &middot; {paper.year}
          </p>
          {truncatedAbstract && (
            <p
              data-testid="hover-abstract"
              className="mt-2 text-xs leading-relaxed text-muted-foreground"
            >
              {truncatedAbstract}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
