"use client";

import type { TrendingTopic } from "@/types/dashboard";

interface TrendingTopicsProps {
  topics: TrendingTopic[];
  loading?: boolean;
}

function getTopicSize(count: number, maxCount: number): string {
  const ratio = count / maxCount;
  if (ratio > 0.75) return "text-base font-semibold";
  if (ratio > 0.5) return "text-sm font-medium";
  return "text-xs font-normal";
}

export function TrendingTopics({ topics, loading }: TrendingTopicsProps) {
  if (loading) {
    return (
      <div data-testid="trending-topics-skeleton" className="flex flex-wrap gap-2">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-7 w-24 animate-pulse rounded-full bg-muted" />
        ))}
      </div>
    );
  }

  if (topics.length === 0) {
    return (
      <div data-testid="trending-topics-empty" className="flex h-32 items-center justify-center text-muted-foreground">
        No trending topics
      </div>
    );
  }

  const maxCount = Math.max(...topics.map((t) => t.count));

  return (
    <div data-testid="trending-topics" className="flex flex-wrap gap-2">
      {topics.map((topic) => (
        <span
          key={topic.topic}
          className={`inline-flex items-center gap-1 rounded-full border border-border bg-secondary/50 px-3 py-1 text-secondary-foreground ${getTopicSize(topic.count, maxCount)}`}
        >
          {topic.topic}
          <span className="ml-1 text-[10px] text-muted-foreground">({topic.count})</span>
        </span>
      ))}
    </div>
  );
}
