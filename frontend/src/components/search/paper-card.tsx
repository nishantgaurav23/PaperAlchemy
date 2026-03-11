"use client";

import Link from "next/link";
import { ExternalLink } from "lucide-react";
import type { Paper } from "@/types/paper";

interface PaperCardProps {
  paper: Paper;
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function formatAuthors(authors: string[]): string {
  if (authors.length === 0) return "Unknown";
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} et al.`;
}

function truncateAbstract(abstract: string, maxLen = 200): string {
  if (abstract.length <= maxLen) return abstract;
  return abstract.slice(0, maxLen) + "...";
}

export function PaperCard({ paper }: PaperCardProps) {
  return (
    <article className="rounded-lg border border-border bg-card p-4 transition-colors hover:bg-accent/50">
      <div className="flex flex-col gap-2">
        <div className="flex items-start justify-between gap-2">
          <Link
            href={`/papers/${paper.id}`}
            className="text-lg font-semibold leading-tight text-foreground hover:text-primary hover:underline"
          >
            {paper.title}
          </Link>
          <a
            href={`https://arxiv.org/abs/${paper.arxiv_id}`}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="arXiv"
            className="shrink-0 text-muted-foreground hover:text-foreground"
          >
            <ExternalLink className="size-4" />
          </a>
        </div>

        <p className="text-sm text-muted-foreground">
          {formatAuthors(paper.authors)}
        </p>

        {paper.abstract && (
          <p
            data-testid="paper-abstract"
            className="text-sm leading-relaxed text-muted-foreground"
          >
            {truncateAbstract(paper.abstract)}
          </p>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <time className="text-xs text-muted-foreground">
            {formatDate(paper.published_date)}
          </time>
          {paper.categories.map((cat) => (
            <span
              key={cat}
              className="inline-flex h-5 items-center rounded-full bg-secondary px-2 text-xs font-medium text-secondary-foreground"
            >
              {cat}
            </span>
          ))}
        </div>
      </div>
    </article>
  );
}
