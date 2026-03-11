"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ExportButton } from "@/components/export";
import { PaperList } from "@/components/collections/paper-list";
import {
  getCollection,
  removePaper,
  reorderPapers,
  generateShareLink,
} from "@/lib/collections";

export default function CollectionDetailPage() {
  const params = useParams();
  const collectionId = params.id as string;
  const [collection, setCollection] = useState(() =>
    getCollection(collectionId)
  );
  const [copied, setCopied] = useState(false);

  function handleRemove(paperId: string) {
    const updated = removePaper(collectionId, paperId);
    setCollection({ ...updated });
  }

  function handleReorder(fromIndex: number, toIndex: number) {
    const updated = reorderPapers(collectionId, fromIndex, toIndex);
    setCollection({ ...updated });
  }

  async function handleShare() {
    const link = generateShareLink(collectionId);
    const fullUrl = `${window.location.origin}${link}`;
    try {
      await navigator.clipboard.writeText(fullUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback: show the link in an alert
      window.prompt("Copy this link:", fullUrl);
    }
  }

  if (!collection) {
    return (
      <div className="flex flex-col gap-4">
        <Link
          href="/collections"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
          aria-label="Back to collections"
        >
          <ArrowLeft className="size-4" />
          Back to collections
        </Link>
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-16 text-center">
          <p className="text-muted-foreground">Collection not found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center gap-2">
        <Link
          href="/collections"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
          aria-label="Back to collections"
        >
          <ArrowLeft className="size-4" />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold tracking-tight">
            {collection.name}
          </h1>
          {collection.description && (
            <p className="text-sm text-muted-foreground">
              {collection.description}
            </p>
          )}
        </div>
        {collection.papers.length > 0 && (
          <ExportButton papers={collection.papers} label="Export All" />
        )}
        <Button variant="outline" size="sm" onClick={handleShare}>
          <Share2 className="size-3.5" data-icon="inline-start" />
          {copied ? "Copied!" : "Share"}
        </Button>
      </div>

      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <span>
          {collection.papers.length}{" "}
          {collection.papers.length === 1 ? "paper" : "papers"}
        </span>
      </div>

      <PaperList
        papers={collection.papers}
        onRemove={handleRemove}
        onReorder={handleReorder}
      />
    </div>
  );
}
