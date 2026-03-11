"use client";

import { ExternalLink } from "lucide-react";
import type { ChatSource } from "@/types/chat";

interface SourceCardProps {
  source: ChatSource;
  index: number;
}

function formatAuthors(authors: string[]): string {
  if (authors.length === 0) return "Unknown";
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} et al.`;
}

export function SourceCard({ source, index }: SourceCardProps) {
  return (
    <div
      data-testid={`source-card-${index + 1}`}
      className="flex items-start gap-3 rounded-lg border border-border p-3"
    >
      <span className="inline-flex size-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
        {index + 1}
      </span>
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <a
          href={`https://arxiv.org/abs/${source.arxiv_id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm font-medium leading-tight text-foreground hover:text-primary hover:underline"
        >
          {source.title}
          <ExternalLink className="ml-1 inline size-3" />
        </a>
        <p className="text-xs text-muted-foreground">
          {formatAuthors(source.authors)} ({source.year})
        </p>
      </div>
    </div>
  );
}
