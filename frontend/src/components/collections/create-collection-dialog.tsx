"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface CreateCollectionDialogProps {
  open: boolean;
  onClose: () => void;
  onCreate: (name: string, description: string) => void;
  initialName?: string;
  initialDescription?: string;
  title?: string;
}

export function CreateCollectionDialog({
  open,
  onClose,
  onCreate,
  initialName = "",
  initialDescription = "",
  title = "New Collection",
}: CreateCollectionDialogProps) {
  const [name, setName] = useState(initialName);
  const [description, setDescription] = useState(initialDescription);

  if (!open) return null;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    onCreate(name.trim(), description.trim());
    setName("");
    setDescription("");
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 shadow-lg">
        <h2 className="text-lg font-semibold">{title}</h2>
        <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-3">
          <div>
            <label
              htmlFor="collection-name"
              className="text-sm font-medium text-foreground"
            >
              Name
            </label>
            <Input
              id="collection-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., ML Papers"
              maxLength={100}
              autoFocus
            />
          </div>
          <div>
            <label
              htmlFor="collection-desc"
              className="text-sm font-medium text-foreground"
            >
              Description (optional)
            </label>
            <Input
              id="collection-desc"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What's this collection about?"
            />
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={!name.trim()}>
              Create
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
