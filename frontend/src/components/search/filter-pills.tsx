"use client";

import { X } from "lucide-react";
import { SORT_OPTIONS } from "@/types/paper";

interface FilterPillsProps {
  query: string;
  category: string;
  sort: string;
  onRemoveCategory: () => void;
  onRemoveSort: () => void;
  onClearQuery: () => void;
}

function Pill({
  label,
  ariaLabel,
  onRemove,
}: {
  label: string;
  ariaLabel: string;
  onRemove: () => void;
}) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
      {label}
      <button
        type="button"
        onClick={onRemove}
        aria-label={ariaLabel}
        className="ml-0.5 inline-flex size-4 items-center justify-center rounded-full hover:bg-primary/20"
      >
        <X className="size-3" />
      </button>
    </span>
  );
}

export function FilterPills({
  query,
  category,
  sort,
  onRemoveCategory,
  onRemoveSort,
  onClearQuery,
}: FilterPillsProps) {
  const hasQuery = query.trim().length > 0;
  const hasCategory = category.length > 0;
  const hasNonDefaultSort = sort !== "relevance" && sort !== "";

  if (!hasQuery && !hasCategory && !hasNonDefaultSort) return null;

  const sortLabel =
    SORT_OPTIONS.find((o) => o.value === sort)?.label ?? sort;

  return (
    <div className="flex flex-wrap items-center gap-2" data-testid="filter-pills">
      <span className="text-xs text-muted-foreground">Active filters:</span>
      {hasQuery && (
        <Pill
          label={`"${query}"`}
          ariaLabel="Remove query filter"
          onRemove={onClearQuery}
        />
      )}
      {hasCategory && (
        <Pill
          label={category}
          ariaLabel="Remove category filter"
          onRemove={onRemoveCategory}
        />
      )}
      {hasNonDefaultSort && (
        <Pill
          label={sortLabel}
          ariaLabel="Remove sort filter"
          onRemove={onRemoveSort}
        />
      )}
    </div>
  );
}
