import type { Paper } from "./paper";

export interface Collection {
  id: string;
  name: string;
  description: string;
  papers: Paper[];
  createdAt: string;
  updatedAt: string;
}

export interface ShareData {
  name: string;
  description: string;
  paperIds: string[];
}

export const STORAGE_KEY = "paperalchemy-collections";
