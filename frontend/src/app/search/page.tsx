"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { SearchBar } from "@/components/search/search-bar";
import { CategoryFilter } from "@/components/search/category-filter";
import { SortSelect } from "@/components/search/sort-select";
import { SearchResults } from "@/components/search/search-results";
import { searchPapers } from "@/lib/api/search";
import type { Paper } from "@/types/paper";
import { PAGE_SIZE } from "@/types/paper";

export default function SearchPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const query = searchParams.get("q") ?? "";
  const category = searchParams.get("category") ?? "";
  const sort = searchParams.get("sort") ?? "relevance";
  const page = Number(searchParams.get("page") ?? "1");

  const [papers, setPapers] = useState<Paper[]>([]);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  function updateParams(updates: Record<string, string>) {
    const params = new URLSearchParams(searchParams.toString());
    for (const [key, value] of Object.entries(updates)) {
      if (value) {
        params.set(key, value);
      } else {
        params.delete(key);
      }
    }
    router.push(`/search?${params.toString()}`);
  }

  const fetchResults = useCallback(async () => {
    if (!query && !category) return;

    setIsLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const data = await searchPapers({
        q: query || undefined,
        category: category || undefined,
        sort,
        page,
      });
      setPapers(data.papers);
      setTotal(data.total);
      setTotalPages(data.total_pages);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setPapers([]);
      setTotal(0);
      setTotalPages(0);
    } finally {
      setIsLoading(false);
    }
  }, [query, category, sort, page]);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  function handleSearch(q: string) {
    updateParams({ q, page: "" });
  }

  function handleClear() {
    updateParams({ q: "", category: "", sort: "", page: "" });
    setPapers([]);
    setTotal(0);
    setTotalPages(0);
    setHasSearched(false);
  }

  function handleCategoryChange(cat: string) {
    updateParams({ category: cat, page: "" });
  }

  function handleSortChange(s: string) {
    updateParams({ sort: s, page: "" });
  }

  function handlePageChange(p: number) {
    updateParams({ page: String(p) });
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold tracking-tight">Search Papers</h1>
        <p className="text-sm text-muted-foreground">
          Discover academic papers by title, author, or topic.
        </p>
      </div>

      <div className="flex flex-col gap-4">
        <SearchBar value={query} onSearch={handleSearch} onClear={handleClear} />

        <div className="flex flex-wrap items-center gap-3">
          <CategoryFilter value={category} onChange={handleCategoryChange} />
          <SortSelect value={sort} onChange={handleSortChange} />
        </div>
      </div>

      <SearchResults
        papers={papers}
        total={total}
        page={page}
        totalPages={totalPages}
        pageSize={PAGE_SIZE}
        isLoading={isLoading}
        error={error}
        hasSearched={hasSearched}
        onPageChange={handlePageChange}
        onRetry={fetchResults}
      />
    </div>
  );
}
