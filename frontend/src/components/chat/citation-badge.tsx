"use client";

interface CitationBadgeProps {
  number: number;
  onClick?: () => void;
}

export function CitationBadge({ number, onClick }: CitationBadgeProps) {
  return (
    <button
      onClick={onClick}
      aria-label={`Citation ${number}`}
      className="inline-flex size-5 items-center justify-center rounded-full bg-primary/10 align-super text-[10px] font-bold text-primary transition-colors hover:bg-primary/20"
    >
      {number}
    </button>
  );
}
