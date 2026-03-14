"use client";

import { useState, useRef, useEffect, type FormEvent } from "react";
import { Search, X, Clock, Trash2 } from "lucide-react";
import { useRecentSearches } from "@/lib/hooks/use-recent-searches";

interface SearchBarProps {
  value: string;
  onSearch: (query: string) => void;
  onClear: () => void;
}

export function SearchBar({ value, onSearch, onClear }: SearchBarProps) {
  const [inputValue, setInputValue] = useState(value);
  const [showDropdown, setShowDropdown] = useState(false);
  const { searches, addSearch, removeSearch, clearAll } = useRecentSearches();
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Sync inputValue when the URL-driven value prop changes
  useEffect(() => {
    setInputValue(value);
  }, [value]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = inputValue.trim();
    if (trimmed) {
      addSearch(trimmed);
    }
    onSearch(trimmed);
    setShowDropdown(false);
  }

  function handleSelectRecent(term: string) {
    setInputValue(term);
    addSearch(term);
    onSearch(term);
    setShowDropdown(false);
  }

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filteredSearches = inputValue.trim()
    ? searches.filter((s) =>
        s.toLowerCase().includes(inputValue.toLowerCase())
      )
    : searches;

  return (
    <div ref={wrapperRef} className="relative w-full">
      <form onSubmit={handleSubmit} className="relative w-full">
        <Search
          data-testid="search-icon"
          className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
        />
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onFocus={() => setShowDropdown(true)}
          placeholder="Search papers by title, author, or topic..."
          className="h-10 w-full rounded-lg border border-input bg-transparent pl-10 pr-10 text-sm outline-none transition-colors placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        />
        {inputValue && (
          <button
            type="button"
            onClick={() => {
              setInputValue("");
              setShowDropdown(false);
              onClear();
            }}
            aria-label="Clear search"
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
          >
            <X className="size-4" />
          </button>
        )}
      </form>

      {showDropdown && filteredSearches.length > 0 && (
        <div
          data-testid="recent-searches-dropdown"
          className="absolute z-50 mt-1 w-full rounded-lg border border-border bg-popover p-1 shadow-lg"
        >
          <div className="flex items-center justify-between px-2 py-1.5">
            <span className="text-xs font-medium text-muted-foreground">
              Recent searches
            </span>
            <button
              type="button"
              onClick={() => {
                clearAll();
                setShowDropdown(false);
              }}
              aria-label="Clear all recent searches"
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              Clear all
            </button>
          </div>
          {filteredSearches.slice(0, 5).map((term) => (
            <div
              key={term}
              className="flex items-center justify-between rounded-md px-2 py-1.5 hover:bg-accent/50"
            >
              <button
                type="button"
                onClick={() => handleSelectRecent(term)}
                className="flex flex-1 items-center gap-2 text-left text-sm"
              >
                <Clock className="size-3.5 text-muted-foreground" />
                <span>{term}</span>
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  removeSearch(term);
                }}
                aria-label={`Remove "${term}" from recent searches`}
                className="text-muted-foreground hover:text-foreground"
              >
                <Trash2 className="size-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
