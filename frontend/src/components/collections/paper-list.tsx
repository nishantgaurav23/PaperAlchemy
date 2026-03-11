"use client";

import { useRef, useState } from "react";
import { ExternalLink, GripVertical, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Paper } from "@/types/paper";

interface PaperListProps {
  papers: Paper[];
  onRemove: (paperId: string) => void;
  onReorder: (fromIndex: number, toIndex: number) => void;
}

export function PaperList({ papers, onRemove, onReorder }: PaperListProps) {
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [dropIndex, setDropIndex] = useState<number | null>(null);
  const dragRef = useRef<number | null>(null);

  if (papers.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <p className="text-muted-foreground">
          No papers in this collection yet. Add papers from search or paper
          detail pages.
        </p>
      </div>
    );
  }

  const showDragHandles = papers.length > 1;

  function handleDragStart(index: number) {
    dragRef.current = index;
    setDragIndex(index);
  }

  function handleDragOver(e: React.DragEvent, index: number) {
    e.preventDefault();
    setDropIndex(index);
  }

  function handleDrop(index: number) {
    if (dragRef.current !== null && dragRef.current !== index) {
      onReorder(dragRef.current, index);
    }
    dragRef.current = null;
    setDragIndex(null);
    setDropIndex(null);
  }

  function handleDragEnd() {
    dragRef.current = null;
    setDragIndex(null);
    setDropIndex(null);
  }

  return (
    <div className="flex flex-col gap-2">
      {papers.map((paper, index) => (
        <div
          key={paper.id}
          className={`flex items-start gap-3 rounded-lg border border-border bg-card p-3 transition-colors ${
            dropIndex === index ? "border-primary bg-accent/50" : ""
          } ${dragIndex === index ? "opacity-50" : ""}`}
          draggable={showDragHandles}
          onDragStart={() => handleDragStart(index)}
          onDragOver={(e) => handleDragOver(e, index)}
          onDrop={() => handleDrop(index)}
          onDragEnd={handleDragEnd}
        >
          {showDragHandles && (
            <div
              className="mt-1 cursor-grab text-muted-foreground"
              aria-label="Drag to reorder"
            >
              <GripVertical className="size-4" />
            </div>
          )}

          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between gap-2">
              <h4 className="font-medium text-foreground leading-tight">
                {paper.title}
              </h4>
              <div className="flex shrink-0 items-center gap-1">
                <a
                  href={`https://arxiv.org/abs/${paper.arxiv_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="arXiv"
                  className="text-muted-foreground hover:text-foreground"
                >
                  <ExternalLink className="size-3.5" />
                </a>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label="Remove paper"
                  onClick={() => onRemove(paper.id)}
                >
                  <X className="size-3.5" />
                </Button>
              </div>
            </div>
            <p className="mt-0.5 text-sm text-muted-foreground">
              {paper.authors.join(", ") || "Unknown"}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}
