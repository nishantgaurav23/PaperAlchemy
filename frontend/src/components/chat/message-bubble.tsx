"use client";

import { type ReactNode } from "react";
import { AlertCircle } from "lucide-react";
import { CitationBadge } from "./citation-badge";
import { SourceCard } from "./source-card";
import type { ChatMessage } from "@/types/chat";

interface MessageBubbleProps {
  message: ChatMessage;
  onRetry?: () => void;
}

function formatTimestamp(ts: number): string {
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 10) return "just now";
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
}

function renderContentWithCitations(content: string): ReactNode[] {
  // Split on citation references like [1], [2], etc.
  const parts = content.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      return <CitationBadge key={i} number={parseInt(match[1], 10)} />;
    }
    return <span key={i}>{part}</span>;
  });
}

function renderMarkdown(content: string): ReactNode[] {
  // Simple markdown: bold, italic, lists, code blocks
  const lines = content.split("\n");
  const result: ReactNode[] = [];

  let inCodeBlock = false;
  let codeLines: string[] = [];
  let listItems: ReactNode[] = [];
  let inList = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Code blocks
    if (line.startsWith("```")) {
      if (inCodeBlock) {
        result.push(
          <pre key={`code-${i}`} className="my-2 overflow-x-auto rounded-md bg-muted p-3 text-xs">
            <code>{codeLines.join("\n")}</code>
          </pre>,
        );
        codeLines = [];
        inCodeBlock = false;
      } else {
        if (inList) {
          result.push(<ul key={`list-${i}`} className="my-1 list-disc pl-5 text-sm">{listItems}</ul>);
          listItems = [];
          inList = false;
        }
        inCodeBlock = true;
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    // List items
    if (line.startsWith("- ") || line.startsWith("* ")) {
      inList = true;
      const itemContent = line.slice(2);
      listItems.push(
        <li key={`li-${i}`}>{renderContentWithCitations(renderInline(itemContent))}</li>,
      );
      continue;
    }

    // Numbered list items
    if (/^\d+\.\s/.test(line)) {
      if (!inList) inList = true;
      const itemContent = line.replace(/^\d+\.\s/, "");
      listItems.push(
        <li key={`li-${i}`}>{renderContentWithCitations(renderInline(itemContent))}</li>,
      );
      continue;
    }

    // Flush list
    if (inList) {
      result.push(
        <ul key={`list-${i}`} className="my-1 list-disc pl-5 text-sm">
          {listItems}
        </ul>,
      );
      listItems = [];
      inList = false;
    }

    // Empty line
    if (line.trim() === "") {
      continue;
    }

    // Regular paragraph
    result.push(
      <p key={`p-${i}`} className="text-sm leading-relaxed">
        {renderContentWithCitations(renderInline(line))}
      </p>,
    );
  }

  // Flush remaining list
  if (inList && listItems.length > 0) {
    result.push(
      <ul key="list-end" className="my-1 list-disc pl-5 text-sm">
        {listItems}
      </ul>,
    );
  }

  return result;
}

function renderInline(text: string): string {
  // Keep as string for now — bold/italic handled by CSS or further processing
  return text
    .replace(/\*\*(.+?)\*\*/g, "$1") // Strip bold markers (rendered via content)
    .replace(/`(.+?)`/g, "$1"); // Strip inline code markers
}

export function MessageBubble({ message, onRetry }: MessageBubbleProps) {
  if (message.role === "error") {
    return (
      <div className="flex justify-center px-4 py-2">
        <div className="flex items-center gap-3 rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3">
          <AlertCircle className="size-4 shrink-0 text-destructive" />
          <p className="text-sm text-destructive">{message.content}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="shrink-0 rounded-md bg-destructive px-3 py-1 text-xs font-medium text-destructive-foreground transition-colors hover:bg-destructive/80"
            >
              Retry
            </button>
          )}
        </div>
      </div>
    );
  }

  const isUser = message.role === "user";

  return (
    <div
      data-testid={`message-${message.id}`}
      className={`flex ${isUser ? "justify-end" : "justify-start"} px-4 py-1`}
    >
      <div
        className={`flex max-w-[85%] flex-col gap-1 rounded-2xl px-4 py-2.5 ${
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground"
        }`}
      >
        <div className="flex flex-col gap-1.5">
          {isUser ? (
            <p className="text-sm">{message.content}</p>
          ) : (
            renderMarkdown(message.content)
          )}
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 flex flex-col gap-2 border-t border-border/50 pt-2">
            <p className="text-xs font-semibold text-muted-foreground">Sources:</p>
            {message.sources.map((source, i) => (
              <SourceCard key={source.arxiv_id} source={source} index={i} />
            ))}
          </div>
        )}

        <time className="text-[10px] opacity-50">{formatTimestamp(message.timestamp)}</time>
      </div>
    </div>
  );
}
