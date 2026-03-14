import { renderHook, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { useRecentSearches } from "./use-recent-searches";

const STORAGE_KEY = "paperalchemy:recent-searches";

// Create a simple localStorage mock since Node 25 built-in localStorage
// doesn't expose standard Web Storage API methods in test environment
function createStorageMock() {
  const store = new Map<string, string>();
  return {
    getItem: vi.fn((key: string) => store.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => store.set(key, value)),
    removeItem: vi.fn((key: string) => store.delete(key)),
    clear: vi.fn(() => store.clear()),
    get length() {
      return store.size;
    },
    key: vi.fn((index: number) => [...store.keys()][index] ?? null),
  };
}

describe("useRecentSearches", () => {
  let storageMock: ReturnType<typeof createStorageMock>;

  beforeEach(() => {
    storageMock = createStorageMock();
    vi.stubGlobal("localStorage", storageMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns empty array when no recent searches", () => {
    const { result } = renderHook(() => useRecentSearches());
    expect(result.current.searches).toEqual([]);
  });

  it("adds a search term", () => {
    const { result } = renderHook(() => useRecentSearches());
    act(() => result.current.addSearch("transformers"));
    expect(result.current.searches).toEqual(["transformers"]);
  });

  it("adds most recent search at the beginning (LIFO)", () => {
    const { result } = renderHook(() => useRecentSearches());
    act(() => result.current.addSearch("attention"));
    act(() => result.current.addSearch("bert"));
    expect(result.current.searches[0]).toBe("bert");
    expect(result.current.searches[1]).toBe("attention");
  });

  it("deduplicates — moves existing term to front", () => {
    const { result } = renderHook(() => useRecentSearches());
    act(() => result.current.addSearch("attention"));
    act(() => result.current.addSearch("bert"));
    act(() => result.current.addSearch("attention"));
    expect(result.current.searches).toEqual(["attention", "bert"]);
  });

  it("caps at max 10 items", () => {
    const { result } = renderHook(() => useRecentSearches());
    for (let i = 0; i < 12; i++) {
      act(() => result.current.addSearch(`query-${i}`));
    }
    expect(result.current.searches).toHaveLength(10);
    expect(result.current.searches[0]).toBe("query-11");
  });

  it("removes a specific search term", () => {
    const { result } = renderHook(() => useRecentSearches());
    act(() => result.current.addSearch("a"));
    act(() => result.current.addSearch("b"));
    act(() => result.current.removeSearch("a"));
    expect(result.current.searches).toEqual(["b"]);
  });

  it("clears all searches", () => {
    const { result } = renderHook(() => useRecentSearches());
    act(() => result.current.addSearch("a"));
    act(() => result.current.addSearch("b"));
    act(() => result.current.clearAll());
    expect(result.current.searches).toEqual([]);
  });

  it("persists to localStorage", () => {
    const { result } = renderHook(() => useRecentSearches());
    act(() => result.current.addSearch("transformers"));
    expect(storageMock.setItem).toHaveBeenCalledWith(
      STORAGE_KEY,
      JSON.stringify(["transformers"])
    );
  });

  it("loads from localStorage on mount", () => {
    storageMock.getItem.mockReturnValueOnce(JSON.stringify(["saved-query"]));
    const { result } = renderHook(() => useRecentSearches());
    expect(result.current.searches).toEqual(["saved-query"]);
  });

  it("ignores empty or whitespace-only terms", () => {
    const { result } = renderHook(() => useRecentSearches());
    act(() => result.current.addSearch(""));
    act(() => result.current.addSearch("   "));
    expect(result.current.searches).toEqual([]);
  });

  it("handles corrupted localStorage gracefully", () => {
    storageMock.getItem.mockReturnValueOnce("not-json");
    const { result } = renderHook(() => useRecentSearches());
    expect(result.current.searches).toEqual([]);
  });
});
