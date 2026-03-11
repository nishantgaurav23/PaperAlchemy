import type { Collection, ShareData } from "@/types/collection";
import type { Paper } from "@/types/paper";
import { STORAGE_KEY } from "@/types/collection";

function readStorage(): Collection[] {
  if (typeof window === "undefined") return [];
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw) as Collection[];
  } catch {
    return [];
  }
}

function writeStorage(collections: Collection[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(collections));
}

function generateId(): string {
  return crypto.randomUUID();
}

export function getCollections(): Collection[] {
  return readStorage().sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  );
}

export function getCollection(id: string): Collection | undefined {
  return readStorage().find((c) => c.id === id);
}

export function createCollection(name: string, description = ""): Collection {
  const trimmed = name.trim();
  if (!trimmed) throw new Error("Collection name cannot be empty");
  if (trimmed.length > 100)
    throw new Error("Collection name must be 100 characters or less");

  const now = new Date().toISOString();
  const collection: Collection = {
    id: generateId(),
    name: trimmed,
    description,
    papers: [],
    createdAt: now,
    updatedAt: now,
  };

  const collections = readStorage();
  collections.push(collection);
  writeStorage(collections);
  return collection;
}

export function updateCollection(
  id: string,
  updates: { name?: string; description?: string }
): Collection {
  const collections = readStorage();
  const index = collections.findIndex((c) => c.id === id);
  if (index === -1) throw new Error(`Collection ${id} not found`);

  if (updates.name !== undefined) {
    const trimmed = updates.name.trim();
    if (!trimmed) throw new Error("Collection name cannot be empty");
    if (trimmed.length > 100)
      throw new Error("Collection name must be 100 characters or less");
    collections[index].name = trimmed;
  }

  if (updates.description !== undefined) {
    collections[index].description = updates.description;
  }

  collections[index].updatedAt = new Date().toISOString();
  writeStorage(collections);
  return collections[index];
}

export function deleteCollection(id: string): void {
  const collections = readStorage();
  const index = collections.findIndex((c) => c.id === id);
  if (index === -1) throw new Error(`Collection ${id} not found`);
  collections.splice(index, 1);
  writeStorage(collections);
}

export function addPaper(collectionId: string, paper: Paper): Collection {
  const collections = readStorage();
  const index = collections.findIndex((c) => c.id === collectionId);
  if (index === -1) throw new Error(`Collection ${collectionId} not found`);

  const exists = collections[index].papers.some((p) => p.id === paper.id);
  if (!exists) {
    collections[index].papers.push(paper);
    collections[index].updatedAt = new Date().toISOString();
    writeStorage(collections);
  }

  return collections[index];
}

export function removePaper(
  collectionId: string,
  paperId: string
): Collection {
  const collections = readStorage();
  const index = collections.findIndex((c) => c.id === collectionId);
  if (index === -1) throw new Error(`Collection ${collectionId} not found`);

  collections[index].papers = collections[index].papers.filter(
    (p) => p.id !== paperId
  );
  collections[index].updatedAt = new Date().toISOString();
  writeStorage(collections);
  return collections[index];
}

export function reorderPapers(
  collectionId: string,
  fromIndex: number,
  toIndex: number
): Collection {
  const collections = readStorage();
  const index = collections.findIndex((c) => c.id === collectionId);
  if (index === -1) throw new Error(`Collection ${collectionId} not found`);

  const papers = [...collections[index].papers];
  const [moved] = papers.splice(fromIndex, 1);
  papers.splice(toIndex, 0, moved);
  collections[index].papers = papers;
  collections[index].updatedAt = new Date().toISOString();
  writeStorage(collections);
  return collections[index];
}

export function generateShareLink(collectionId: string): string {
  const collection = getCollection(collectionId);
  if (!collection) throw new Error(`Collection ${collectionId} not found`);

  const shareData: ShareData = {
    name: collection.name,
    description: collection.description,
    paperIds: collection.papers.map((p) => p.id),
  };

  const encoded = btoa(JSON.stringify(shareData));
  return `/collections/shared?data=${encoded}`;
}

export function parseShareLink(link: string): ShareData | undefined {
  try {
    const url = new URL(link, "http://localhost");
    const data = url.searchParams.get("data");
    if (!data) return undefined;
    return JSON.parse(atob(data)) as ShareData;
  } catch {
    return undefined;
  }
}
