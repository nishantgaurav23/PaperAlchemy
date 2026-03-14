"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2, Globe, Database } from "lucide-react";
import { SearchBar } from "@/components/search/search-bar";
import { CategoryFilter } from "@/components/search/category-filter";
import { SortSelect } from "@/components/search/sort-select";
import { FilterPills } from "@/components/search/filter-pills";
import { MobileFilterSheet } from "@/components/search/mobile-filter-sheet";
import { SearchResults } from "@/components/search/search-results";
import { PullToRefresh } from "@/components/layout/pull-to-refresh";
import { searchPapers, searchArxiv } from "@/lib/api/search";
import type { Paper, SearchHit } from "@/types/paper";
import { PAGE_SIZE } from "@/types/paper";

/** Convert a backend SearchHit to the Paper shape used by UI components. */
function hitToPaper(hit: SearchHit): Paper {
  return {
    id: hit.arxiv_id,
    arxiv_id: hit.arxiv_id,
    title: hit.title,
    authors: hit.authors,
    abstract: hit.abstract,
    categories: [],
    published_date: "",
    pdf_url: hit.pdf_url,
  };
}

type SearchMode = "knowledge_base" | "arxiv";

function SearchContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const query = searchParams.get("q") ?? "";
  const category = searchParams.get("category") ?? "";
  const sort = searchParams.get("sort") ?? "relevance";
  const page = Number(searchParams.get("page") ?? "1");
  const modeParam = searchParams.get("mode") ?? "arxiv";

  const [searchMode, setSearchMode] = useState<SearchMode>(
    modeParam === "knowledge_base" ? "knowledge_base" : "arxiv",
  );
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
      if (searchMode === "arxiv") {
        // Live arXiv web search
        const data = await searchArxiv({
          query: query || category || "",
          category: category || undefined,
          maxResults: PAGE_SIZE,
          sortBy: sort,
        });

        const papers: Paper[] = data.hits.map((h) => ({
          id: h.arxiv_id,
          arxiv_id: h.arxiv_id,
          title: h.title,
          authors: h.authors,
          abstract: h.abstract,
          categories: h.categories,
          published_date: h.published_date,
          pdf_url: h.pdf_url,
        }));

        setPapers(papers);
        setTotal(data.total);
        setTotalPages(1); // arXiv returns a single page
      } else {
        // Knowledge base search
        const data = await searchPapers({
          q: query || undefined,
          category: category || undefined,
          sort,
          page,
        });
        const seen = new Set<string>();
        const unique = data.hits.filter((h) => {
          if (seen.has(h.arxiv_id)) return false;
          seen.add(h.arxiv_id);
          return true;
        });
        setPapers(unique.map(hitToPaper));
        setTotal(data.total);
        setTotalPages(data.size > 0 ? Math.ceil(data.total / data.size) : 0);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setPapers([]);
      setTotal(0);
      setTotalPages(0);
    } finally {
      setIsLoading(false);
    }
  }, [query, category, sort, page, searchMode]);

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

  function handleRemoveCategory() {
    updateParams({ category: "", page: "" });
  }

  function handleRemoveSort() {
    updateParams({ sort: "", page: "" });
  }

  function handleClearQuery() {
    updateParams({ q: "", page: "" });
  }

  function handleSuggestionClick(suggestion: string) {
    updateParams({ q: suggestion, page: "" });
  }

  function handleModeChange(mode: SearchMode) {
    setSearchMode(mode);
    updateParams({ mode, page: "" });
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold tracking-tight">Search Papers</h1>
        <p className="text-sm text-muted-foreground">
          {searchMode === "arxiv"
            ? "Search arXiv for papers online in real-time."
            : "Search papers in the local knowledge base."}
        </p>
      </div>

      {/* Search mode toggle */}
      <div className="flex items-center gap-1 rounded-lg border border-border p-1 w-fit">
        <button
          type="button"
          onClick={() => handleModeChange("arxiv")}
          className={`inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
            searchMode === "arxiv"
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground hover:bg-accent"
          }`}
        >
          <Globe className="size-4" />
          arXiv Online
        </button>
        <button
          type="button"
          onClick={() => handleModeChange("knowledge_base")}
          className={`inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
            searchMode === "knowledge_base"
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground hover:bg-accent"
          }`}
        >
          <Database className="size-4" />
          Knowledge Base
        </button>
      </div>

      <div className="flex flex-col gap-4">
        <SearchBar value={query} onSearch={handleSearch} onClear={handleClear} />

        <div className="flex flex-wrap items-center gap-3">
          <div className="hidden md:contents">
            <CategoryFilter value={category} onChange={handleCategoryChange} />
            <SortSelect value={sort} onChange={handleSortChange} />
          </div>
          <MobileFilterSheet
            category={category}
            sort={sort}
            onCategoryChange={handleCategoryChange}
            onSortChange={handleSortChange}
          />
        </div>

        <FilterPills
          query={query}
          category={category}
          sort={sort}
          onRemoveCategory={handleRemoveCategory}
          onRemoveSort={handleRemoveSort}
          onClearQuery={handleClearQuery}
        />
      </div>

      <PullToRefresh onRefresh={async () => { await fetchResults(); }}>
        <SearchResults
          papers={papers}
          total={total}
          page={page}
          totalPages={totalPages}
          pageSize={PAGE_SIZE}
          isLoading={isLoading}
          error={error}
          hasSearched={hasSearched}
          searchQuery={query}
          onPageChange={handlePageChange}
          onRetry={fetchResults}
          onSuggestionClick={handleSuggestionClick}
        />
      </PullToRefresh>
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center py-16">
          <Loader2 className="size-8 animate-spin text-muted-foreground" />
        </div>
      }
    >
      <SearchContent />
    </Suspense>
  );
}
