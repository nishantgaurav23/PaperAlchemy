"use client";

import { useState } from "react";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CollectionCard } from "@/components/collections/collection-card";
import { CreateCollectionDialog } from "@/components/collections/create-collection-dialog";
import {
  getCollections,
  createCollection,
  deleteCollection,
} from "@/lib/collections";

export default function CollectionsPage() {
  const [collections, setCollections] = useState(getCollections);
  const [showCreate, setShowCreate] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  function handleCreate(name: string, description: string) {
    createCollection(name, description);
    setCollections(getCollections());
    setShowCreate(false);
  }

  function handleDeleteConfirm() {
    if (!deleteTarget) return;
    deleteCollection(deleteTarget);
    setCollections(getCollections());
    setDeleteTarget(null);
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Collections</h1>
          <p className="text-sm text-muted-foreground">
            Organize your papers into reading lists.
          </p>
        </div>
        <Button onClick={() => setShowCreate(true)}>
          <Plus className="size-4" data-icon="inline-start" />
          New Collection
        </Button>
      </div>

      {collections.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border py-16 text-center">
          <p className="text-muted-foreground">
            Create your first collection to start organizing papers.
          </p>
          <Button className="mt-4" onClick={() => setShowCreate(true)}>
            <Plus className="size-4" data-icon="inline-start" />
            Create your first collection
          </Button>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {collections.map((col) => (
            <CollectionCard
              key={col.id}
              collection={col}
              onDelete={setDeleteTarget}
            />
          ))}
        </div>
      )}

      <CreateCollectionDialog
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onCreate={handleCreate}
      />

      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-sm rounded-lg border border-border bg-card p-6 shadow-lg">
            <h2 className="text-lg font-semibold">Delete Collection?</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              This will remove the collection but not the papers themselves.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <Button variant="ghost" onClick={() => setDeleteTarget(null)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={handleDeleteConfirm}>
                Delete
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
