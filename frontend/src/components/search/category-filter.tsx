"use client";

import { ARXIV_CATEGORIES } from "@/types/paper";

interface CategoryFilterProps {
  value: string;
  onChange: (category: string) => void;
}

export function CategoryFilter({ value, onChange }: CategoryFilterProps) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor="category-filter" className="sr-only">
        Category
      </label>
      <select
        id="category-filter"
        aria-label="Category"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-9 rounded-lg border border-input bg-transparent px-2.5 text-sm outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
      >
        <option value="">All Categories</option>
        {ARXIV_CATEGORIES.map((cat) => (
          <option key={cat.value} value={cat.value}>
            {cat.value}
          </option>
        ))}
      </select>
    </div>
  );
}
