"use client";

import { useEffect, useRef, useState } from "react";

interface MermaidBlockProps {
  children: string;
}

/**
 * Sanitize Mermaid syntax to fix common LLM generation issues.
 * LLMs frequently generate syntax that Mermaid's parser rejects:
 *
 * 1. Parentheses inside square bracket node labels:
 *    `A[Encoder (N layers)]` → `A["Encoder (N layers)"]`
 *
 * 2. Parentheses in subgraph titles:
 *    `subgraph Encoder Block (per layer)` → `subgraph Encoder Block - per layer`
 *
 * 3. Edge labels with commas (parsed as node separators):
 *    `A -- K, V from Encoder --> B` → `A -- "K, V from Encoder" --> B`
 */
function sanitizeMermaid(code: string): string {
  return code
    .split("\n")
    .map((line) => {
      const trimmed = line.trimStart();

      // Fix subgraph titles containing parentheses
      // `subgraph Encoder Block (per layer)` → `subgraph Encoder Block - per layer`
      if (trimmed.startsWith("subgraph ")) {
        return line.replace(/\(([^)]*)\)/g, "- $1");
      }

      // Fix edge labels containing commas: -- label with, commas -->
      // Must quote the label to prevent Mermaid treating commas as separators
      const edgeLabelMatch = line.match(
        /^(\s*\S+\s+--\s+)([^">[{]+[,][^">[{]*?)(-->|---)(\s*.*)$/,
      );
      if (edgeLabelMatch) {
        const [, prefix, label, arrow, suffix] = edgeLabelMatch;
        return `${prefix}"${label.trim()}" ${arrow}${suffix}`;
      }

      return line;
    })
    .join("\n")
    .replace(
      // Fix node labels with [] containing () but not already quoted
      /(\w+)\[([^\]"]*\([^\]]*\))\]/g,
      (_, id: string, label: string) => `${id}["${label}"]`,
    );
}

export function MermaidBlock({ children }: MermaidBlockProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function renderDiagram() {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: document.documentElement.classList.contains("dark")
            ? "dark"
            : "default",
          fontFamily: "inherit",
          securityLevel: "strict",
        });

        const sanitized = sanitizeMermaid(children.trim());
        const id = `mermaid-${Math.random().toString(36).slice(2, 9)}`;
        const { svg } = await mermaid.render(id, sanitized);

        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg;
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to render diagram");
        }
      }
    }

    renderDiagram();
    return () => {
      cancelled = true;
    };
  }, [children]);

  if (error) {
    return (
      <div className="my-2 rounded-lg border border-destructive/50 bg-destructive/10 p-3">
        <p className="text-xs text-destructive">Diagram render error: {error}</p>
        <pre className="mt-2 overflow-x-auto text-xs text-muted-foreground">
          <code>{children}</code>
        </pre>
      </div>
    );
  }

  return (
    <div className="my-3 flex justify-center overflow-x-auto rounded-lg border border-border bg-card p-4">
      <div ref={containerRef} className="mermaid-diagram" />
    </div>
  );
}
