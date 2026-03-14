"use client";

import { Search, AlertCircle, FileSearch, Sparkles } from "lucide-react";
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
  searchQuery?: string;
  onPageChange: (page: number) => void;
  onRetry?: () => void;
  onSuggestionClick?: (query: string) => void;
}

const SEARCH_SUGGESTIONS = [
  "transformer architecture",
  "reinforcement learning",
  "large language models",
  "computer vision",
  "graph neural networks",
];

function ShimmerSkeleton() {
  return (
    <div
      data-testid="paper-skeleton"
      className="glass-card overflow-hidden p-4"
    >
      <div className="flex flex-col gap-3">
        <div className="shimmer h-5 w-3/4 rounded" />
        <div className="shimmer h-4 w-1/3 rounded" />
        <div className="shimmer h-4 w-full rounded" />
        <div className="shimmer h-4 w-2/3 rounded" />
        <div className="flex gap-2">
          <div className="shimmer h-5 w-16 rounded-full" />
          <div className="shimmer h-5 w-12 rounded-full" />
        </div>
      </div>
    </div>
  );
}

function LoadingSkeletons() {
  return (
    <div className="flex flex-col gap-4">
      {Array.from({ length: 4 }).map((_, i) => (
        <ShimmerSkeleton key={i} />
      ))}
    </div>
  );
}

function EmptyState({
  searchQuery,
  onSuggestionClick,
}: {
  searchQuery?: string;
  onSuggestionClick?: (query: string) => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-6 py-16 text-center">
      <div className="relative">
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 blur-xl" />
        <div className="relative flex size-20 items-center justify-center rounded-full bg-gradient-to-br from-primary/10 to-accent/10">
          <FileSearch className="size-10 text-muted-foreground" />
        </div>
      </div>
      <div className="flex flex-col gap-2">
        <h3 className="text-lg font-semibold">No papers found</h3>
        <p className="max-w-sm text-sm text-muted-foreground">
          {searchQuery
            ? `No results for "${searchQuery.length > 50 ? searchQuery.slice(0, 50) + "..." : searchQuery}". Try a different query or broader terms.`
            : "Try adjusting your search query or filters."}
        </p>
      </div>
      {onSuggestionClick && (
        <div className="flex flex-col items-center gap-2">
          <span className="flex items-center gap-1 text-xs text-muted-foreground">
            <Sparkles className="size-3" />
            Try searching for:
          </span>
          <div className="flex flex-wrap justify-center gap-2">
            {SEARCH_SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => onSuggestionClick(suggestion)}
                className="rounded-full border border-border bg-card px-3 py-1 text-xs text-foreground transition-colors hover:border-primary hover:bg-primary/5 hover:text-primary"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function InitialState() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
      <div className="relative">
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 blur-xl" />
        <div className="relative flex size-20 items-center justify-center rounded-full bg-gradient-to-br from-primary/10 to-accent/10">
          <Search className="size-10 text-muted-foreground" />
        </div>
      </div>
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
  searchQuery,
  onPageChange,
  onRetry,
  onSuggestionClick,
}: SearchResultsProps) {
  if (isLoading) return <LoadingSkeletons />;
  if (error) return <ErrorState onRetry={onRetry} />;
  if (!hasSearched) return <InitialState />;
  if (papers.length === 0)
    return (
      <EmptyState
        searchQuery={searchQuery}
        onSuggestionClick={onSuggestionClick}
      />
    );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3">
        {papers.map((paper, index) => (
          <PaperCard
            key={paper.id}
            paper={paper}
            animationDelay={Math.min(index, 10) * 50}
          />
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
