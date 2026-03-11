"use client";

import { Search, AlertCircle, FileSearch } from "lucide-react";
import { PaperCard } from "./paper-card";
import { Pagination } from "./pagination";
import type { Paper } from "@/types/paper";

interface SearchResultsProps {
  papers: Paper[];
  total: number;
  page: number;
  totalPages: number;
  pageSize: number;
  isLoading: boolean;
  error: string | null;
  hasSearched?: boolean;
  onPageChange: (page: number) => void;
  onRetry?: () => void;
}

function LoadingSkeletons() {
  return (
    <div className="flex flex-col gap-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <div
          key={i}
          data-testid="paper-skeleton"
          className="animate-pulse rounded-lg border border-border bg-card p-4"
        >
          <div className="flex flex-col gap-3">
            <div className="h-5 w-3/4 rounded bg-muted" />
            <div className="h-4 w-1/3 rounded bg-muted" />
            <div className="h-4 w-full rounded bg-muted" />
            <div className="h-4 w-2/3 rounded bg-muted" />
            <div className="flex gap-2">
              <div className="h-5 w-16 rounded-full bg-muted" />
              <div className="h-5 w-12 rounded-full bg-muted" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
      <FileSearch className="size-12 text-muted-foreground" />
      <div className="flex flex-col gap-1">
        <h3 className="text-lg font-semibold">No papers found</h3>
        <p className="text-sm text-muted-foreground">
          Try adjusting your search query or filters.
        </p>
      </div>
    </div>
  );
}

function InitialState() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
      <Search className="size-12 text-muted-foreground" />
      <div className="flex flex-col gap-1">
        <h3 className="text-lg font-semibold">Search for papers</h3>
        <p className="text-sm text-muted-foreground">
          Enter a query to search by title, author, or topic.
        </p>
      </div>
    </div>
  );
}

function ErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
      <AlertCircle className="size-12 text-destructive" />
      <div className="flex flex-col gap-1">
        <h3 className="text-lg font-semibold">Something went wrong</h3>
        <p className="text-sm text-muted-foreground">
          Failed to fetch search results. Please try again.
        </p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex h-9 items-center justify-center rounded-lg bg-primary px-4 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/80"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function SearchResults({
  papers,
  total,
  page,
  totalPages,
  pageSize,
  isLoading,
  error,
  hasSearched = false,
  onPageChange,
  onRetry,
}: SearchResultsProps) {
  if (isLoading) return <LoadingSkeletons />;
  if (error) return <ErrorState onRetry={onRetry} />;
  if (!hasSearched) return <InitialState />;
  if (papers.length === 0) return <EmptyState />;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3">
        {papers.map((paper) => (
          <PaperCard key={paper.id} paper={paper} />
        ))}
      </div>

      {totalPages > 1 && (
        <Pagination
          currentPage={page}
          totalPages={totalPages}
          totalResults={total}
          pageSize={pageSize}
          onPageChange={onPageChange}
        />
      )}
    </div>
  );
}
