"use client";

import { useState } from "react";
import { Bookmark, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { getCollections, addPaper, createCollection } from "@/lib/collections";
import type { Paper } from "@/types/paper";

interface AddToCollectionProps {
  paper: Paper;
  onAdded?: (collectionId: string) => void;
}

export function AddToCollection({ paper, onAdded }: AddToCollectionProps) {
  const [open, setOpen] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");

  const collections = getCollections();

  function handleAdd(collectionId: string) {
    addPaper(collectionId, paper);
    setOpen(false);
    onAdded?.(collectionId);
  }

  function handleCreate() {
    if (!newName.trim()) return;
    const col = createCollection(newName.trim());
    addPaper(col.id, paper);
    setNewName("");
    setShowCreate(false);
    setOpen(false);
    onAdded?.(col.id);
  }

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="icon-xs"
        aria-label="Add to collection"
        onClick={() => setOpen(!open)}
      >
        <Bookmark className="size-3.5" />
      </Button>

      {open && (
        <div className="absolute right-0 top-full z-50 mt-1 w-56 rounded-lg border border-border bg-popover p-1 shadow-md">
          {collections.length === 0 && !showCreate && (
            <p className="px-2 py-1.5 text-sm text-muted-foreground">
              No collections yet
            </p>
          )}

          {collections.map((col) => (
            <button
              key={col.id}
              className="flex w-full items-center rounded-md px-2 py-1.5 text-sm hover:bg-accent"
              onClick={() => handleAdd(col.id)}
            >
              {col.name}
            </button>
          ))}

          {showCreate ? (
            <div className="flex items-center gap-1 p-1">
              <input
                type="text"
                className="h-7 flex-1 rounded border border-input bg-transparent px-2 text-sm"
                placeholder="Collection name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                autoFocus
              />
              <Button variant="ghost" size="icon-xs" onClick={handleCreate}>
                <Plus className="size-3.5" />
              </Button>
            </div>
          ) : (
            <button
              className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-sm text-primary hover:bg-accent"
              onClick={() => setShowCreate(true)}
            >
              <Plus className="size-3.5" />
              Create new
            </button>
          )}
        </div>
      )}
    </div>
  );
}
