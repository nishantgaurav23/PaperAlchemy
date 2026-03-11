"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PaperSection } from "@/types/paper";

interface PaperSectionsProps {
  sections: PaperSection[] | undefined;
}

export function PaperSections({ sections }: PaperSectionsProps) {
  const [expanded, setExpanded] = useState<Set<number>>(() => {
    if (!sections || sections.length === 0) return new Set();
    // First 2 sections expanded by default
    const initial = new Set<number>();
    for (let i = 0; i < Math.min(2, sections.length); i++) {
      initial.add(i);
    }
    return initial;
  });

  if (!sections || sections.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
        Sections not yet parsed. Upload or re-index this paper to extract sections.
      </div>
    );
  }

  const toggleSection = (index: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  return (
    <div className="flex flex-col gap-1">
      <h2 className="mb-2 text-lg font-semibold">Sections</h2>
      {sections.map((section, index) => {
        const isExpanded = expanded.has(index);
        return (
          <div key={index} className="rounded-lg border border-border">
            <button
              role="button"
              onClick={() => toggleSection(index)}
              className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-medium hover:bg-accent/50"
              aria-expanded={isExpanded}
            >
              {section.title}
              <ChevronDown
                className={cn(
                  "size-4 shrink-0 transition-transform",
                  isExpanded && "rotate-180",
                )}
              />
            </button>
            <div
              data-state={isExpanded ? "expanded" : "collapsed"}
              className={cn(
                "overflow-hidden transition-all",
                isExpanded ? "px-4 pb-4" : "h-0",
              )}
            >
              <p className="text-sm leading-relaxed text-muted-foreground">
                {section.content}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
