"use client";

import Link from "next/link";
import type { Paper } from "@/types/paper";
import { Badge } from "@/components/ui/badge";

interface RelatedPapersProps {
  papers: Paper[] | undefined;
}

function formatAuthors(authors: string[]): string {
  if (authors.length === 0) return "Unknown";
  if (authors.length <= 2) return authors.join(", ");
  return `${authors.slice(0, 2).join(", ")} et al.`;
}

export function RelatedPapers({ papers }: RelatedPapersProps) {
  if (!papers || papers.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-lg font-semibold">Related Papers</h2>
      <div className="flex gap-4 overflow-x-auto pb-2">
        {papers.map((paper) => (
          <div
            key={paper.id}
            className="flex w-72 shrink-0 flex-col gap-2 rounded-lg border border-border p-4 transition-colors hover:bg-accent/50"
          >
            <Link
              href={`/papers/${paper.id}`}
              className="text-sm font-semibold leading-tight hover:text-primary hover:underline"
            >
              {paper.title}
            </Link>
            <p className="text-xs text-muted-foreground">
              {formatAuthors(paper.authors)}
            </p>
            <div className="flex flex-wrap gap-1">
              {paper.categories.map((cat) => (
                <Badge key={cat} variant="secondary" className="text-xs">
                  {cat}
                </Badge>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
