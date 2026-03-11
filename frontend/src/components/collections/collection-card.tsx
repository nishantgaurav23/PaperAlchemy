"use client";

import Link from "next/link";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Collection } from "@/types/collection";

interface CollectionCardProps {
  collection: Collection;
  onDelete: (id: string) => void;
}

export function CollectionCard({ collection, onDelete }: CollectionCardProps) {
  const paperCount = collection.papers.length;

  return (
    <div className="rounded-lg border border-border bg-card p-4 transition-colors hover:bg-accent/50">
      <div className="flex items-start justify-between gap-2">
        <Link
          href={`/collections/${collection.id}`}
          className="flex-1 min-w-0"
        >
          <h3 className="font-semibold text-foreground truncate">
            {collection.name}
          </h3>
          {collection.description && (
            <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
              {collection.description}
            </p>
          )}
          <p className="mt-2 text-xs text-muted-foreground">
            {paperCount} {paperCount === 1 ? "paper" : "papers"}
          </p>
        </Link>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label="Delete collection"
          onClick={(e) => {
            e.preventDefault();
            onDelete(collection.id);
          }}
        >
          <Trash2 className="size-3.5" />
        </Button>
      </div>
    </div>
  );
}
