"use client";

import { SORT_OPTIONS } from "@/types/paper";

interface SortSelectProps {
  value: string;
  onChange: (sort: string) => void;
}

export function SortSelect({ value, onChange }: SortSelectProps) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor="sort-select" className="sr-only">
        Sort by
      </label>
      <select
        id="sort-select"
        aria-label="Sort by"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-9 rounded-lg border border-input bg-transparent px-2.5 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
      >
        {SORT_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
