"use client";

import { ExternalLink, FileText, Globe, Database } from "lucide-react";
import type { ChatSource } from "@/types/chat";

interface SourceCardProps {
  source: ChatSource;
  index: number;
}

function formatAuthors(authors: string[]): string {
  if (authors.length === 0) return "";
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} et al.`;
}

function getSourceUrl(source: ChatSource): string {
  if (source.arxiv_url) return source.arxiv_url;
  if (source.arxiv_id) return `https://arxiv.org/abs/${source.arxiv_id}`;
  if (source.url) return source.url;
  return "";
}

function SourceTypeBadge({ type }: { type?: string }) {
  switch (type) {
    case "arxiv":
      return (
        <span className="inline-flex items-center gap-0.5 rounded-full bg-orange-100 px-1.5 py-0.5 text-[10px] font-medium text-orange-700 dark:bg-orange-900/40 dark:text-orange-300">
          <Globe className="size-2.5" />
          arXiv
        </span>
      );
    case "web":
      return (
        <span className="inline-flex items-center gap-0.5 rounded-full bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 dark:bg-blue-900/40 dark:text-blue-300">
          <Globe className="size-2.5" />
          Web
        </span>
      );
    case "knowledge_base":
    default:
      return (
        <span className="inline-flex items-center gap-0.5 rounded-full bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300">
          <Database className="size-2.5" />
          KB
        </span>
      );
  }
}

export function SourceCard({ source, index }: SourceCardProps) {
  const url = getSourceUrl(source);
  const authorsStr = formatAuthors(source.authors);

  return (
    <div
      data-testid={`source-card-${index + 1}`}
      className="group flex items-start gap-3 rounded-lg border border-border bg-background/50 p-3 transition-all hover:border-primary/30 hover:shadow-sm"
    >
      <span className="inline-flex size-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-accent/20 text-xs font-bold text-primary">
        {index + 1}
      </span>
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <div className="flex items-start gap-1.5">
          <FileText className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
          {url ? (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium leading-tight text-foreground transition-colors group-hover:text-primary"
            >
              {source.title}
              <ExternalLink className="ml-1 inline size-3 opacity-0 transition-opacity group-hover:opacity-100" />
            </a>
          ) : (
            <span className="text-sm font-medium leading-tight text-foreground">
              {source.title}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 pl-5">
          <SourceTypeBadge type={source.source_type} />
          {authorsStr && (
            <p className="text-xs text-muted-foreground">
              {authorsStr}{source.year ? ` (${source.year})` : ""}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
