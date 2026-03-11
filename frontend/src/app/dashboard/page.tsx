"use client";

import { useEffect, useState } from "react";
import { BarChart3 } from "lucide-react";
import { getDashboardData } from "@/lib/api/dashboard";
import { StatsCards } from "@/components/dashboard/stats-cards";
import { CategoryChart } from "@/components/dashboard/category-chart";
import { TimelineChart } from "@/components/dashboard/timeline-chart";
import { HotPapers } from "@/components/dashboard/hot-papers";
import { TrendingTopics } from "@/components/dashboard/trending-topics";
import type { DashboardData } from "@/types/dashboard";

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [usingMock, setUsingMock] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function fetchData() {
      setLoading(true);
      try {
        const result = await getDashboardData();
        if (!cancelled) {
          setData(result);
          // If the API call fell back to mock data (getDashboardData catches errors internally),
          // we check if we got mock data by trying the real API again
          try {
            const { apiClient } = await import("@/lib/api-client");
            await apiClient.get("/api/v1/dashboard/stats");
          } catch {
            setUsingMock(true);
          }
        }
      } catch {
        if (!cancelled) {
          const result = await getDashboardData(true);
          setData(result);
          setUsingMock(true);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchData();
    return () => { cancelled = true; };
  }, []);

  return (
    <div data-testid="dashboard-page" className="space-y-6 p-6">
      <div className="flex items-center gap-3">
        <BarChart3 className="size-7 text-primary" />
        <div>
          <h1 className="text-2xl font-bold text-foreground">Research Trends</h1>
          <p className="text-sm text-muted-foreground">Analytics and insights from your paper collection</p>
        </div>
      </div>

      {usingMock && (
        <div data-testid="mock-data-banner" className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
          Using sample data — connect backend API for live analytics
        </div>
      )}

      <StatsCards stats={data?.stats ?? null} loading={loading} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <section className="rounded-lg border border-border bg-card p-4">
          <h2 className="mb-4 text-lg font-semibold text-foreground">Category Breakdown</h2>
          <CategoryChart categories={data?.categories ?? []} loading={loading} />
        </section>

        <section className="rounded-lg border border-border bg-card p-4">
          <h2 className="mb-4 text-lg font-semibold text-foreground">Publication Timeline</h2>
          <TimelineChart timeline={data?.timeline ?? []} loading={loading} />
        </section>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <section className="rounded-lg border border-border bg-card p-4">
          <h2 className="mb-4 text-lg font-semibold text-foreground">Hot Papers</h2>
          <HotPapers papers={data?.hot_papers ?? []} loading={loading} />
        </section>

        <section className="rounded-lg border border-border bg-card p-4">
          <h2 className="mb-4 text-lg font-semibold text-foreground">Trending Topics</h2>
          <TrendingTopics topics={data?.trending_topics ?? []} loading={loading} />
        </section>
      </div>
    </div>
  );
}
