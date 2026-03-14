"use client";

import { useState } from "react";
import { SlidersHorizontal, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { ARXIV_CATEGORIES, SORT_OPTIONS } from "@/types/paper";
import { cn } from "@/lib/utils";

interface MobileFilterSheetProps {
  category: string;
  sort: string;
  onCategoryChange: (category: string) => void;
  onSortChange: (sort: string) => void;
}

export function MobileFilterSheet({
  category,
  sort,
  onCategoryChange,
  onSortChange,
}: MobileFilterSheetProps) {
  const [open, setOpen] = useState(false);

  const activeCount =
    (category ? 1 : 0) + (sort && sort !== "relevance" ? 1 : 0);

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen(true)}
        aria-label="Filters"
        className="relative min-h-[44px] gap-2 md:hidden"
      >
        <SlidersHorizontal className="size-4" />
        Filters
        {activeCount > 0 && (
          <span
            data-testid="filter-count"
            className="flex size-5 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-primary-foreground"
          >
            {activeCount}
          </span>
        )}
      </Button>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="bottom" className="max-h-[80vh] overflow-y-auto rounded-t-2xl">
          <SheetHeader>
            <SheetTitle>Filters</SheetTitle>
            <SheetDescription>Filter and sort search results</SheetDescription>
          </SheetHeader>

          <div className="mt-6 space-y-6">
            {/* Category */}
            <div>
              <h3 className="mb-3 text-sm font-semibold text-foreground">Category</h3>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => {
                    onCategoryChange("");
                    setOpen(false);
                  }}
                  aria-label="All categories"
                  className={cn(
                    "min-h-[44px] rounded-lg border px-3 py-2 text-sm transition-colors",
                    !category
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border text-muted-foreground hover:border-primary/50"
                  )}
                >
                  All
                </button>
                {ARXIV_CATEGORIES.map((cat) => (
                  <button
                    key={cat.value}
                    type="button"
                    onClick={() => {
                      onCategoryChange(cat.value);
                      setOpen(false);
                    }}
                    aria-label={cat.value}
                    className={cn(
                      "min-h-[44px] rounded-lg border px-3 py-2 text-sm transition-colors",
                      category === cat.value
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-border text-muted-foreground hover:border-primary/50"
                    )}
                  >
                    <span className="flex items-center gap-1.5">
                      {category === cat.value && <Check className="size-3.5" />}
                      {cat.value}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Sort */}
            <div>
              <h3 className="mb-3 text-sm font-semibold text-foreground">Sort by</h3>
              <div className="flex flex-col gap-1">
                {SORT_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => {
                      onSortChange(option.value);
                      setOpen(false);
                    }}
                    aria-label={option.label}
                    className={cn(
                      "min-h-[44px] flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-left transition-colors",
                      sort === option.value
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-muted-foreground hover:bg-accent/50"
                    )}
                  >
                    {sort === option.value && <Check className="size-4" />}
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </SheetContent>
      </Sheet>
    </>
  );
}
