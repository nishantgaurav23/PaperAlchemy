"use client";

import { useState, type FormEvent } from "react";
import { Search, X } from "lucide-react";

interface SearchBarProps {
  value: string;
  onSearch: (query: string) => void;
  onClear: () => void;
}

export function SearchBar({ value, onSearch, onClear }: SearchBarProps) {
  const [inputValue, setInputValue] = useState(value);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    onSearch(inputValue.trim());
  }

  return (
    <form onSubmit={handleSubmit} className="relative w-full">
      <Search
        data-testid="search-icon"
        className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
      />
      <input
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        placeholder="Search papers by title, author, or topic..."
        className="h-10 w-full rounded-lg border border-input bg-transparent pl-10 pr-10 text-sm outline-none transition-colors placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
      />
      {value && (
        <button
          type="button"
          onClick={() => {
            setInputValue("");
            onClear();
          }}
          aria-label="Clear search"
          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
        >
          <X className="size-4" />
        </button>
      )}
    </form>
  );
}
