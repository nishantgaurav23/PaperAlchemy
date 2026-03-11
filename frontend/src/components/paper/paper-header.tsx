"use client";

import { useState } from "react";
import { ExternalLink, FileDown, Copy, Check } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExportButton } from "@/components/export";
import type { PaperDetail } from "@/types/paper";

interface PaperHeaderProps {
  paper: PaperDetail;
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

function buildCitation(paper: PaperDetail): string {
  const authors = paper.authors.join(", ");
  const year = paper.published_date ? new Date(paper.published_date).getFullYear() : "";
  return `${authors}. "${paper.title}." arXiv preprint arXiv:${paper.arxiv_id} (${year}). https://arxiv.org/abs/${paper.arxiv_id}`;
}

export function PaperHeader({ paper }: PaperHeaderProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyCitation = () => {
    navigator.clipboard.writeText(buildCitation(paper));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-2xl font-bold leading-tight md:text-3xl">
        {paper.title}
      </h1>

      {paper.authors.length > 0 && (
        <p className="text-sm text-muted-foreground">
          {paper.authors.join(", ")}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-3">
        {paper.published_date && (
          <time className="text-sm text-muted-foreground">
            {formatDate(paper.published_date)}
          </time>
        )}

        {paper.categories.map((cat) => (
          <Badge key={cat} variant="secondary">
            {cat}
          </Badge>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <a
          href={`https://arxiv.org/abs/${paper.arxiv_id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
        >
          arXiv
          <ExternalLink className="size-3" />
        </a>

        {paper.pdf_url && (
          <a
            href={paper.pdf_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
          >
            PDF
            <FileDown className="size-3" />
          </a>
        )}

        <Button
          variant="outline"
          size="sm"
          onClick={handleCopyCitation}
          aria-label="Copy citation"
        >
          {copied ? (
            <Check className="size-3.5" />
          ) : (
            <Copy className="size-3.5" />
          )}
          {copied ? "Copied" : "Copy Citation"}
        </Button>

        <ExportButton papers={[paper]} />
      </div>

      {paper.abstract && (
        <p className="text-sm leading-relaxed text-muted-foreground">
          {paper.abstract}
        </p>
      )}
    </div>
  );
}
