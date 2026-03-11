"use client";

import Link from "next/link";
import { ExternalLink } from "lucide-react";
import type { Paper } from "@/types/paper";

interface HotPapersProps {
  papers: Paper[];
  loading?: boolean;
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function formatAuthors(authors: string[]): string {
  if (authors.length === 0) return "Unknown";
  if (authors.length <= 2) return authors.join(", ");
  return `${authors[0]} et al.`;
}

export function HotPapers({ papers, loading }: HotPapersProps) {
  if (loading) {
    return (
      <div data-testid="hot-papers-skeleton" className="space-y-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="h-20 animate-pulse rounded-lg bg-muted" />
        ))}
      </div>
    );
  }

  if (papers.length === 0) {
    return (
      <div data-testid="hot-papers-empty" className="flex h-40 items-center justify-center text-muted-foreground">
        No papers found
      </div>
    );
  }

  return (
    <div data-testid="hot-papers" className="space-y-3">
      {papers.map((paper) => (
        <article key={paper.id} className="rounded-lg border border-border bg-card p-3 transition-colors hover:bg-accent/50">
          <div className="flex items-start justify-between gap-2">
            <Link
              href={`/papers/${paper.id}`}
              className="line-clamp-1 text-sm font-semibold text-foreground hover:text-primary hover:underline"
            >
              {paper.title}
            </Link>
            <a
              href={`https://arxiv.org/abs/${paper.arxiv_id}`}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`arXiv link for ${paper.title}`}
              className="shrink-0 text-muted-foreground hover:text-foreground"
            >
              <ExternalLink className="size-3.5" />
            </a>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">{formatAuthors(paper.authors)}</p>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <time className="text-xs text-muted-foreground">{formatDate(paper.published_date)}</time>
            {paper.categories.map((cat) => (
              <span
                key={cat}
                className="inline-flex h-4 items-center rounded-full bg-secondary px-1.5 text-[10px] font-medium text-secondary-foreground"
              >
                {cat}
              </span>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}
