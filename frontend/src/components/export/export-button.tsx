"use client";

import { useState, useRef, useEffect } from "react";
import { Download, Check, FileText, Code, Presentation } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  formatBibtex,
  formatMarkdown,
  formatSlideSnippet,
  formatBulkBibtex,
  formatBulkMarkdown,
  copyToClipboard,
  downloadFile,
} from "@/lib/export";
import type { Paper } from "@/types/paper";

interface ExportButtonProps {
  papers: Paper[];
  label?: string;
}

type Format = "bibtex" | "markdown" | "slide";

const FORMAT_CONFIG = [
  { key: "bibtex" as Format, label: "BibTeX", icon: Code, ext: ".bib" },
  { key: "markdown" as Format, label: "Markdown", icon: FileText, ext: ".md" },
  { key: "slide" as Format, label: "Slide Snippet", icon: Presentation, ext: ".txt" },
];

function getExportContent(papers: Paper[], format: Format): string {
  if (papers.length === 0) return "";
  if (papers.length === 1) {
    switch (format) {
      case "bibtex":
        return formatBibtex(papers[0]);
      case "markdown":
        return formatMarkdown(papers[0]);
      case "slide":
        return formatSlideSnippet(papers[0]);
    }
  }
  switch (format) {
    case "bibtex":
      return formatBulkBibtex(papers);
    case "markdown":
      return formatBulkMarkdown(papers);
    case "slide":
      return papers.map(formatSlideSnippet).join("\n\n---\n\n");
  }
}

function getFilename(papers: Paper[], format: Format): string {
  const ext = FORMAT_CONFIG.find((f) => f.key === format)?.ext ?? ".txt";
  if (papers.length === 1 && papers[0].arxiv_id) {
    return `${papers[0].arxiv_id}${ext}`;
  }
  return `papers-export${ext}`;
}

export function ExportButton({ papers, label = "Export" }: ExportButtonProps) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState<Format | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [open]);

  async function handleCopy(format: Format) {
    const content = getExportContent(papers, format);
    const ok = await copyToClipboard(content);
    if (ok) {
      setCopied(format);
      setTimeout(() => setCopied(null), 2000);
    }
  }

  function handleDownload(format: Format) {
    const content = getExportContent(papers, format);
    const filename = getFilename(papers, format);
    downloadFile(content, filename);
    setOpen(false);
  }

  return (
    <div className="relative" ref={menuRef}>
      <Button
        variant="outline"
        size="sm"
        onClick={() => setOpen((prev) => !prev)}
        aria-label={label}
      >
        <Download className="size-3.5" />
        {label}
      </Button>

      {open && (
        <div className="absolute right-0 z-50 mt-1 w-56 rounded-lg border border-border bg-background p-1 shadow-lg">
          {FORMAT_CONFIG.map(({ key, label: formatLabel, icon: Icon }) => (
            <div
              key={key}
              className="flex items-center justify-between rounded-md px-2 py-1.5 text-sm hover:bg-muted"
            >
              <button
                data-action="copy"
                className="flex flex-1 items-center gap-2"
                onClick={() => handleCopy(key)}
              >
                <Icon className="size-3.5 text-muted-foreground" />
                {copied === key ? "Copied!" : formatLabel}
                {copied === key && <Check className="size-3 text-green-600" />}
              </button>
              <button
                aria-label={`Download ${formatLabel}`}
                className="rounded p-1 hover:bg-muted-foreground/10"
                onClick={() => handleDownload(key)}
              >
                <Download className="size-3.5 text-muted-foreground" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
