"use client";

import { useState, useMemo } from "react";
import { ChevronDown, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PaperSection } from "@/types/paper";

interface PaperSectionsProps {
  sections: PaperSection[] | undefined;
}

/** Important top-level sections to show — case-insensitive match */
const IMPORTANT_SECTION_PATTERNS = [
  /^abstract$/i,
  /^introduction$/i,
  /^related\s+work/i,
  /^background/i,
  /^method/i,
  /^approach/i,
  /^model/i,
  /^architecture/i,
  /^experiment/i,
  /^result/i,
  /^evaluation/i,
  /^discussion/i,
  /^conclusion/i,
  /^future\s+work/i,
  /^limitation/i,
  /^contribution/i,
  /^overview/i,
  /^training/i,
  /^inference/i,
  /^pre-training/i,
  /^post-training/i,
  /^fine-tuning/i,
  /^ablation/i,
];

function isImportantSection(title: string): boolean {
  const cleaned = title.replace(/^\d+[\.\)]\s*/, "").trim();
  return IMPORTANT_SECTION_PATTERNS.some((pattern) => pattern.test(cleaned));
}

function getSectionLevel(title: string): number {
  // Detect section numbering depth: "1" = level 1, "1.1" = level 2, "1.1.1" = level 3
  const match = title.match(/^(\d+(?:\.\d+)*)/);
  if (match) {
    return match[1].split(".").length;
  }
  return 1;
}

export function PaperSections({ sections }: PaperSectionsProps) {
  const { importantSections, otherSections } = useMemo(() => {
    if (!sections || sections.length === 0) {
      return { importantSections: [] as PaperSection[], otherSections: [] as PaperSection[] };
    }

    // Filter to top-level or important sections, skip deeply nested subsections
    const important: PaperSection[] = [];
    const other: PaperSection[] = [];

    for (const section of sections) {
      const level = getSectionLevel(section.title);
      // Skip deeply nested sub-subsections (level 3+)
      if (level >= 3) continue;

      // Skip very short sections (likely just headers without content)
      if (!section.content || section.content.trim().length < 50) continue;

      if (isImportantSection(section.title) || level === 1) {
        important.push(section);
      } else {
        other.push(section);
      }
    }

    return { importantSections: important, otherSections: other };
  }, [sections]);

  const [expanded, setExpanded] = useState<Set<string>>(() => {
    // Expand first 3 important sections by default
    const initial = new Set<string>();
    for (let i = 0; i < Math.min(3, importantSections.length); i++) {
      initial.add(`important-${i}`);
    }
    return initial;
  });
  const [showMore, setShowMore] = useState(false);

  if (!sections || sections.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border p-6 text-center text-muted-foreground">
        Sections not yet parsed. Upload or re-index this paper to extract sections.
      </div>
    );
  }

  if (importantSections.length === 0 && otherSections.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border p-6 text-center text-muted-foreground">
        No substantial sections found in this paper.
      </div>
    );
  }

  const toggleSection = (key: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  const renderSection = (section: PaperSection, key: string) => {
    const isExpanded = expanded.has(key);
    const cleanTitle = section.title.replace(/^\d+[\.\)]\s*/, "").trim();

    return (
      <div key={key} className="rounded-xl border border-border bg-card overflow-hidden">
        <button
          role="button"
          onClick={() => toggleSection(key)}
          className="flex w-full items-center justify-between px-5 py-4 text-left font-semibold text-foreground hover:bg-accent/30 transition-colors"
          aria-expanded={isExpanded}
        >
          <span className="text-base">{cleanTitle || section.title}</span>
          <ChevronDown
            className={cn(
              "size-5 shrink-0 text-muted-foreground transition-transform duration-200",
              isExpanded && "rotate-180",
            )}
          />
        </button>
        <div
          data-state={isExpanded ? "expanded" : "collapsed"}
          className={cn(
            "overflow-hidden transition-all duration-200",
            isExpanded ? "px-5 pb-5" : "h-0",
          )}
        >
          <div className="prose prose-sm max-w-none text-foreground/80 leading-relaxed whitespace-pre-line text-[0.95rem]">
            {section.content}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2 mb-1">
        <BookOpen className="size-5 text-primary" />
        <h2 className="text-xl font-bold">Paper Sections</h2>
      </div>

      {/* Important sections */}
      <div className="flex flex-col gap-2">
        {importantSections.map((section, index) =>
          renderSection(section, `important-${index}`)
        )}
      </div>

      {/* Other sections — collapsible */}
      {otherSections.length > 0 && (
        <div className="flex flex-col gap-2">
          <button
            onClick={() => setShowMore(!showMore)}
            className="flex items-center gap-2 text-sm font-medium text-primary hover:text-primary/80 transition-colors py-2"
          >
            <ChevronDown
              className={cn(
                "size-4 transition-transform duration-200",
                showMore && "rotate-180",
              )}
            />
            {showMore ? "Hide" : "Show"} {otherSections.length} more section{otherSections.length !== 1 ? "s" : ""}
          </button>
          {showMore &&
            otherSections.map((section, index) =>
              renderSection(section, `other-${index}`)
            )}
        </div>
      )}
    </div>
  );
}
