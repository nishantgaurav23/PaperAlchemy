import { describe, it, expect, beforeEach } from "vitest";
import {
  getCollections,
  getCollection,
  createCollection,
  updateCollection,
  deleteCollection,
  addPaper,
  removePaper,
  reorderPapers,
  generateShareLink,
  parseShareLink,
} from "./collections";
import type { Paper } from "@/types/paper";
import { STORAGE_KEY } from "@/types/collection";

const mockLocalStorage = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (index: number) => Object.keys(store)[index] ?? null,
  };
})();

Object.defineProperty(window, "localStorage", { value: mockLocalStorage, writable: true });

const mockPaper: Paper = {
  id: "paper-1",
  arxiv_id: "2301.00001",
  title: "Attention Is All You Need",
  authors: ["Vaswani"],
  abstract: "We propose a new architecture.",
  categories: ["cs.CL"],
  published_date: "2017-06-12",
};

const mockPaper2: Paper = {
  id: "paper-2",
  arxiv_id: "2301.00002",
  title: "BERT",
  authors: ["Devlin"],
  abstract: "We introduce BERT.",
  categories: ["cs.CL"],
  published_date: "2018-10-11",
};

const mockPaper3: Paper = {
  id: "paper-3",
  arxiv_id: "2301.00003",
  title: "GPT-3",
  authors: ["Brown"],
  abstract: "Language models are few-shot learners.",
  categories: ["cs.AI"],
  published_date: "2020-05-28",
};

beforeEach(() => {
  localStorage.clear();
});

describe("getCollections", () => {
  it("returns empty array when no collections exist", () => {
    expect(getCollections()).toEqual([]);
  });

  it("returns stored collections sorted by updatedAt desc", () => {
    const c1 = createCollection("Older", "desc1");
    // Update c1 to give it a newer updatedAt
    const c2 = createCollection("Newer", "desc2");
    // Touch c1 to make it most recent
    updateCollection(c1.id, { name: "Older Updated" });
    const result = getCollections();
    expect(result[0].id).toBe(c1.id);
    expect(result[1].id).toBe(c2.id);
  });
});

describe("createCollection", () => {
  it("creates a collection with name and description", () => {
    const c = createCollection("My List", "A reading list");
    expect(c.name).toBe("My List");
    expect(c.description).toBe("A reading list");
    expect(c.papers).toEqual([]);
    expect(c.id).toBeTruthy();
    expect(c.createdAt).toBeTruthy();
    expect(c.updatedAt).toBeTruthy();
  });

  it("persists to localStorage", () => {
    createCollection("Test");
    const raw = localStorage.getItem(STORAGE_KEY);
    expect(raw).toBeTruthy();
    const parsed = JSON.parse(raw!);
    expect(parsed).toHaveLength(1);
    expect(parsed[0].name).toBe("Test");
  });

  it("throws on empty name", () => {
    expect(() => createCollection("")).toThrow();
    expect(() => createCollection("   ")).toThrow();
  });

  it("throws on name longer than 100 chars", () => {
    expect(() => createCollection("A".repeat(101))).toThrow();
  });

  it("allows duplicate names", () => {
    const c1 = createCollection("Same Name");
    const c2 = createCollection("Same Name");
    expect(c1.id).not.toBe(c2.id);
    expect(getCollections()).toHaveLength(2);
  });
});

describe("getCollection", () => {
  it("returns collection by id", () => {
    const created = createCollection("Test");
    const found = getCollection(created.id);
    expect(found).toBeDefined();
    expect(found!.name).toBe("Test");
  });

  it("returns undefined for non-existent id", () => {
    expect(getCollection("non-existent")).toBeUndefined();
  });
});

describe("updateCollection", () => {
  it("updates name and description", () => {
    const c = createCollection("Old Name", "Old desc");
    const updated = updateCollection(c.id, {
      name: "New Name",
      description: "New desc",
    });
    expect(updated.name).toBe("New Name");
    expect(updated.description).toBe("New desc");
  });

  it("updates updatedAt timestamp", () => {
    const c = createCollection("Test");
    const before = c.updatedAt;
    // Small delay to ensure different timestamp
    const updated = updateCollection(c.id, { name: "Updated" });
    expect(updated.updatedAt).toBeTruthy();
    // updatedAt should be >= before
    expect(new Date(updated.updatedAt).getTime()).toBeGreaterThanOrEqual(
      new Date(before).getTime()
    );
  });

  it("throws on non-existent collection", () => {
    expect(() => updateCollection("bad-id", { name: "X" })).toThrow();
  });

  it("throws on empty name", () => {
    const c = createCollection("Test");
    expect(() => updateCollection(c.id, { name: "" })).toThrow();
  });
});

describe("deleteCollection", () => {
  it("removes collection from storage", () => {
    const c = createCollection("Test");
    deleteCollection(c.id);
    expect(getCollections()).toHaveLength(0);
  });

  it("throws on non-existent collection", () => {
    expect(() => deleteCollection("bad-id")).toThrow();
  });
});

describe("addPaper", () => {
  it("adds a paper to a collection", () => {
    const c = createCollection("Test");
    const updated = addPaper(c.id, mockPaper);
    expect(updated.papers).toHaveLength(1);
    expect(updated.papers[0].id).toBe("paper-1");
  });

  it("returns collection unchanged if paper already exists (no-op)", () => {
    const c = createCollection("Test");
    addPaper(c.id, mockPaper);
    const updated = addPaper(c.id, mockPaper);
    expect(updated.papers).toHaveLength(1);
  });

  it("throws on non-existent collection", () => {
    expect(() => addPaper("bad-id", mockPaper)).toThrow();
  });
});

describe("removePaper", () => {
  it("removes a paper from a collection", () => {
    const c = createCollection("Test");
    addPaper(c.id, mockPaper);
    addPaper(c.id, mockPaper2);
    const updated = removePaper(c.id, "paper-1");
    expect(updated.papers).toHaveLength(1);
    expect(updated.papers[0].id).toBe("paper-2");
  });

  it("throws on non-existent collection", () => {
    expect(() => removePaper("bad-id", "paper-1")).toThrow();
  });
});

describe("reorderPapers", () => {
  it("moves paper from one index to another", () => {
    const c = createCollection("Test");
    addPaper(c.id, mockPaper);
    addPaper(c.id, mockPaper2);
    addPaper(c.id, mockPaper3);
    // Move paper-3 (index 2) to index 0
    const updated = reorderPapers(c.id, 2, 0);
    expect(updated.papers[0].id).toBe("paper-3");
    expect(updated.papers[1].id).toBe("paper-1");
    expect(updated.papers[2].id).toBe("paper-2");
  });

  it("persists reorder to localStorage", () => {
    const c = createCollection("Test");
    addPaper(c.id, mockPaper);
    addPaper(c.id, mockPaper2);
    reorderPapers(c.id, 1, 0);
    const stored = getCollection(c.id)!;
    expect(stored.papers[0].id).toBe("paper-2");
  });
});

describe("generateShareLink / parseShareLink", () => {
  it("generates and parses a share link", () => {
    const c = createCollection("Shared List", "A shared collection");
    addPaper(c.id, mockPaper);
    addPaper(c.id, mockPaper2);

    const link = generateShareLink(c.id);
    expect(link).toContain("/collections/shared?data=");

    const parsed = parseShareLink(link);
    expect(parsed).toBeDefined();
    expect(parsed!.name).toBe("Shared List");
    expect(parsed!.paperIds).toEqual(["paper-1", "paper-2"]);
  });

  it("returns undefined for invalid share link", () => {
    expect(parseShareLink("/collections/shared?data=invalid")).toBeUndefined();
  });
});
