"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { SearchBar } from "@/components/search/search-bar";
import {
  FileText,
  AlertCircle,
  Loader2,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Sparkles,
} from "lucide-react";
import { listPapers, requestAnalysis } from "@/lib/api/papers";
import type { Paper, PaperDetail } from "@/types/paper";

const PAGE_SIZE = 20;

function formatAuthors(authors: string[]): string {
  if (authors.length === 0) return "Unknown";
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} et al.`;
}

/** Inline expandable paper card for the Papers tab */
function PaperListItem({ paper }: { paper: Paper }) {
  const [expanded, setExpanded] = useState(false);
  const [highlights, setHighlights] = useState<PaperDetail["highlights"] | null>(null);
  const [isLoadingHighlights, setIsLoadingHighlights] = useState(false);
  const [highlightsError, setHighlightsError] = useState<string | null>(null);

  async function handleToggle() {
    if (expanded) {
      setExpanded(false);
      return;
    }

    setExpanded(true);

    // Only fetch highlights if we haven't loaded them yet
    if (!highlights && !isLoadingHighlights) {
      setIsLoadingHighlights(true);
      setHighlightsError(null);
      try {
        const detail = await requestAnalysis(paper.id);
        setHighlights(detail.highlights ?? null);
      } catch (err) {
        setHighlightsError(
          err instanceof Error ? err.message : "Failed to load highlights",
        );
      } finally {
        setIsLoadingHighlights(false);
      }
    }
  }

  return (
    <article className="rounded-xl border border-border bg-card transition-all duration-200 hover:shadow-md">
      {/* Paper header — always visible */}
      <button
        type="button"
        onClick={handleToggle}
        className="flex w-full items-start gap-3 p-4 text-left"
      >
        <div className="flex flex-1 flex-col gap-1.5 min-w-0">
          <h3 className="text-base font-semibold leading-tight text-foreground">
            {paper.title}
          </h3>
          <p className="text-sm text-muted-foreground">
            {formatAuthors(paper.authors)}
          </p>
          <p className="text-sm leading-relaxed text-foreground/65 line-clamp-2">
            {paper.abstract}
          </p>
          <div className="flex items-center gap-3 mt-1">
            <a
              href={`https://arxiv.org/abs/${paper.arxiv_id}`}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
            >
              <ExternalLink className="size-3" />
              arxiv.org/abs/{paper.arxiv_id}
            </a>
          </div>
        </div>
        <div className="shrink-0 mt-1 text-muted-foreground">
          {expanded ? (
            <ChevronUp className="size-5" />
          ) : (
            <ChevronDown className="size-5" />
          )}
        </div>
      </button>

      {/* Expanded highlights section */}
      {expanded && (
        <div className="border-t border-border px-4 py-4">
          {isLoadingHighlights && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Generating highlights...
            </div>
          )}

          {highlightsError && (
            <p className="text-sm text-destructive">{highlightsError}</p>
          )}

          {!isLoadingHighlights && !highlightsError && !highlights && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Sparkles className="size-4" />
              No highlights available yet.
            </div>
          )}

          {highlights && (
            <div className="flex flex-col gap-4">
              {highlights.novel_contributions.length > 0 && (
                <div>
                  <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Novel Contributions
                  </h4>
                  <ul className="flex flex-col gap-1">
                    {highlights.novel_contributions.map((item, i) => (
                      <li
                        key={i}
                        className="flex gap-2 text-sm leading-relaxed"
                      >
                        <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-primary" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {highlights.important_findings.length > 0 && (
                <div>
                  <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Important Findings
                  </h4>
                  <ul className="flex flex-col gap-1">
                    {highlights.important_findings.map((item, i) => (
                      <li
                        key={i}
                        className="flex gap-2 text-sm leading-relaxed"
                      >
                        <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-emerald-500" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {highlights.practical_implications.length > 0 && (
                <div>
                  <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Practical Implications
                  </h4>
                  <ul className="flex flex-col gap-1">
                    {highlights.practical_implications.map((item, i) => (
                      <li
                        key={i}
                        className="flex gap-2 text-sm leading-relaxed"
                      >
                        <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-amber-500" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </article>
  );
}

function PapersContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const query = searchParams.get("q") ?? "";
  const page = Number(searchParams.get("page") ?? "1");

  const [papers, setPapers] = useState<Paper[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  function updateParams(updates: Record<string, string>) {
    const params = new URLSearchParams(searchParams.toString());
    for (const [key, value] of Object.entries(updates)) {
      if (value) {
        params.set(key, value);
      } else {
        params.delete(key);
      }
    }
    router.push(`/papers?${params.toString()}`);
  }

  const fetchPapers = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const offset = (page - 1) * PAGE_SIZE;
      const data = await listPapers({
        query: query || undefined,
        limit: PAGE_SIZE,
        offset,
      });
      setPapers(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load papers");
      setPapers([]);
    } finally {
      setIsLoading(false);
    }
  }, [query, page]);

  useEffect(() => {
    fetchPapers();
  }, [fetchPapers]);

  function handleSearch(q: string) {
    updateParams({ q, page: "" });
  }

  function handleClear() {
    updateParams({ q: "", page: "" });
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold tracking-tight">Papers</h1>
        <p className="text-sm text-muted-foreground">
          Browse papers in the knowledge base. Click a paper to see AI-generated
          highlights.
        </p>
      </div>

      <SearchBar value={query} onSearch={handleSearch} onClear={handleClear} />

      {isLoading && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="size-8 animate-spin text-muted-foreground" />
        </div>
      )}

      {error && (
        <div className="flex flex-col items-center gap-3 py-16 text-center">
          <AlertCircle className="size-12 text-destructive" />
          <p className="text-sm text-muted-foreground">{error}</p>
          <button
            onClick={fetchPapers}
            className="inline-flex h-9 items-center rounded-lg bg-primary px-4 text-sm font-medium text-primary-foreground hover:bg-primary/80"
          >
            Retry
          </button>
        </div>
      )}

      {!isLoading && !error && papers.length === 0 && (
        <div className="flex flex-col items-center gap-3 py-16 text-center">
          <FileText className="size-12 text-muted-foreground" />
          <h3 className="text-lg font-semibold">No papers yet</h3>
          <p className="text-sm text-muted-foreground">
            Ingest papers from arXiv to get started.
          </p>
        </div>
      )}

      {!isLoading && !error && papers.length > 0 && (
        <div className="flex flex-col gap-3">
          {papers.map((paper) => (
            <PaperListItem key={paper.id} paper={paper} />
          ))}
          {papers.length === PAGE_SIZE && (
            <button
              onClick={() => updateParams({ page: String(page + 1) })}
              className="mx-auto inline-flex h-9 items-center rounded-lg border border-border px-4 text-sm font-medium hover:bg-accent"
            >
              Load more
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default function PapersPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-16">
          <Loader2 className="size-8 animate-spin text-muted-foreground" />
        </div>
      }
    >
      <PapersContent />
    </Suspense>
  );
}
