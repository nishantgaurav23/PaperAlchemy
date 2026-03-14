"use client";

import { type ReactNode, type ComponentPropsWithoutRef } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { AlertCircle } from "lucide-react";
import { CitationBadge } from "./citation-badge";
import { SourceCard } from "./source-card";
import { CodeBlock } from "./code-block";
import { MermaidBlock } from "./mermaid-block";
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

/** Markdown components override for react-markdown */
const markdownComponents = {
  // Custom code block renderer with copy button
  pre({ children, ...props }: ComponentPropsWithoutRef<"pre">) {
    return <div {...props}>{children}</div>;
  },
  code({ children, className, ...props }: ComponentPropsWithoutRef<"code">) {
    // Detect fenced code blocks by the language- class added by rehype-highlight
    const match = /language-(\w+)/.exec(className || "");
    const isBlock = match || (typeof children === "string" && children.includes("\n"));

    if (isBlock) {
      const content = String(children).replace(/\n$/, "");
      // Render mermaid code blocks as diagrams
      if (match?.[1] === "mermaid") {
        return <MermaidBlock>{content}</MermaidBlock>;
      }
      return <CodeBlock language={match?.[1]}>{content}</CodeBlock>;
    }

    // Inline code
    return (
      <code
        className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono text-foreground"
        {...props}
      >
        {children}
      </code>
    );
  },
  // Style paragraph text to include citation badges
  p({ children, ...props }: ComponentPropsWithoutRef<"p">) {
    // Process string children to inject citation badges
    const processed = processChildren(children);
    return (
      <p className="text-base leading-relaxed mb-2" {...props}>
        {processed}
      </p>
    );
  },
  h1({ children, ...props }: ComponentPropsWithoutRef<"h1">) {
    return <h1 className="mt-4 mb-2 text-xl font-bold" {...props}>{children}</h1>;
  },
  h2({ children, ...props }: ComponentPropsWithoutRef<"h2">) {
    return <h2 className="mt-4 mb-2 text-lg font-semibold" {...props}>{children}</h2>;
  },
  h3({ children, ...props }: ComponentPropsWithoutRef<"h3">) {
    return <h3 className="mt-3 mb-1 text-base font-semibold" {...props}>{children}</h3>;
  },
  ul({ children, ...props }: ComponentPropsWithoutRef<"ul">) {
    return <ul className="my-2 list-disc pl-5 text-base" {...props}>{children}</ul>;
  },
  ol({ children, ...props }: ComponentPropsWithoutRef<"ol">) {
    return <ol className="my-2 list-decimal pl-5 text-base" {...props}>{children}</ol>;
  },
  li({ children, ...props }: ComponentPropsWithoutRef<"li">) {
    const processed = processChildren(children);
    return <li className="leading-relaxed" {...props}>{processed}</li>;
  },
  blockquote({ children, ...props }: ComponentPropsWithoutRef<"blockquote">) {
    return (
      <blockquote className="my-3 border-l-3 border-primary/30 pl-4 text-base italic text-muted-foreground" {...props}>
        {children}
      </blockquote>
    );
  },
  table({ children, ...props }: ComponentPropsWithoutRef<"table">) {
    return (
      <div className="my-3 overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm" {...props}>{children}</table>
      </div>
    );
  },
  th({ children, ...props }: ComponentPropsWithoutRef<"th">) {
    return <th className="border-b border-border bg-muted px-3 py-2 text-left text-sm font-semibold" {...props}>{children}</th>;
  },
  td({ children, ...props }: ComponentPropsWithoutRef<"td">) {
    return <td className="border-b border-border/50 px-3 py-2 text-sm" {...props}>{children}</td>;
  },
  a({ children, href, ...props }: ComponentPropsWithoutRef<"a">) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" className="text-primary underline hover:text-primary/80" {...props}>
        {children}
      </a>
    );
  },
  strong({ children, ...props }: ComponentPropsWithoutRef<"strong">) {
    return <strong className="font-semibold" {...props}>{children}</strong>;
  },
};

/** Recursively process children to inject citation badges into text */
function processChildren(children: ReactNode): ReactNode {
  if (typeof children === "string") {
    return renderContentWithCitations(children);
  }
  if (Array.isArray(children)) {
    return children.map((child, i) => {
      if (typeof child === "string") {
        return <span key={i}>{renderContentWithCitations(child)}</span>;
      }
      return child;
    });
  }
  return children;
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
            <p className="text-base">{message.content}</p>
          ) : (
            <div className="prose-chat">
              <Markdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeHighlight, rehypeKatex]}
                components={markdownComponents}
              >
                {message.content}
              </Markdown>
            </div>
          )}
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 flex flex-col gap-2 border-t border-border/50 pt-2">
            <p className="text-xs font-semibold text-muted-foreground">Sources:</p>
            {message.sources.map((source, i) => (
              <SourceCard key={`${source.arxiv_id}-${i}`} source={source} index={i} />
            ))}
          </div>
        )}

        <time className="text-[10px] opacity-50" data-testid="message-timestamp">
          {formatTimestamp(message.timestamp)}
        </time>
      </div>
    </div>
  );
}
