"use client";

import { useState } from "react";
import { ExternalLink, Bookmark } from "lucide-react";
import type { Paper } from "@/types/paper";

interface PaperCardProps {
  paper: Paper;
  /** Animation delay in ms for staggered fade-in */
  animationDelay?: number;
}

/** Map arXiv category prefixes to distinct color classes */
const CATEGORY_COLORS: Record<string, string> = {
  "cs.AI": "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300",
  "cs.CL": "bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300",
  "cs.CV": "bg-sky-100 text-sky-700 dark:bg-sky-900/40 dark:text-sky-300",
  "cs.LG": "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
  "cs.NE": "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  "cs.IR": "bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300",
  "cs.RO": "bg-cyan-100 text-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-300",
  "stat.ML": "bg-fuchsia-100 text-fuchsia-700 dark:bg-fuchsia-900/40 dark:text-fuchsia-300",
  "math.OC": "bg-lime-100 text-lime-700 dark:bg-lime-900/40 dark:text-lime-300",
  "eess.SP": "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300",
};

const DEFAULT_CATEGORY_COLOR =
  "bg-secondary text-secondary-foreground";

function getCategoryColor(cat: string): string {
  return CATEGORY_COLORS[cat] ?? DEFAULT_CATEGORY_COLOR;
}

function formatDate(dateStr: string): string {
  if (!dateStr) return "";
  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return "";
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

export function PaperCard({ paper, animationDelay = 0 }: PaperCardProps) {
  const [bookmarked, setBookmarked] = useState(false);
  const [showAbstract, setShowAbstract] = useState(false);

  return (
    <article
      className="glass-card p-4 transition-all duration-200 hover:shadow-md animate-fade-in-up"
      style={{ animationDelay: `${animationDelay}ms` }}
      onMouseEnter={() => setShowAbstract(true)}
      onMouseLeave={() => setShowAbstract(false)}
    >
      <div className="flex flex-col gap-2">
        <div className="flex items-start justify-between gap-2">
          <a
            href={`https://arxiv.org/abs/${paper.arxiv_id}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-lg font-bold leading-tight text-foreground hover:text-primary hover:underline"
          >
            {paper.title}
          </a>
          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={() => setBookmarked((prev) => !prev)}
              aria-label={bookmarked ? "Remove bookmark" : "Bookmark paper"}
              data-testid="bookmark-button"
              className="text-muted-foreground hover:text-primary"
            >
              <Bookmark
                className={`size-4 ${bookmarked ? "fill-primary text-primary" : ""}`}
              />
            </button>
            <a
              href={`https://arxiv.org/abs/${paper.arxiv_id}`}
              target="_blank"
              rel="noopener noreferrer"
              aria-label="arXiv"
              className="text-muted-foreground hover:text-foreground"
            >
              <ExternalLink className="size-4" />
            </a>
          </div>
        </div>

        <p className="text-sm font-medium text-foreground/70">
          {formatAuthors(paper.authors)}
        </p>

        {paper.abstract && (
          <p
            data-testid="paper-abstract"
            className={`text-sm leading-relaxed text-foreground/65 transition-all duration-200 ${
              showAbstract ? "line-clamp-none" : "line-clamp-2"
            }`}
          >
            {showAbstract ? paper.abstract : truncateAbstract(paper.abstract)}
          </p>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <time className="text-xs text-muted-foreground">
            {formatDate(paper.published_date)}
          </time>
          {paper.categories.map((cat) => (
            <span
              key={cat}
              data-testid="category-chip"
              className={`inline-flex h-5 items-center rounded-full px-2 text-xs font-medium ${getCategoryColor(cat)}`}
            >
              {cat}
            </span>
          ))}
        </div>
      </div>
    </article>
  );
}
