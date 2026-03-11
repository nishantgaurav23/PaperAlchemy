"use client";

import { FileText, TrendingUp, FolderOpen, Star } from "lucide-react";
import type { DashboardStats } from "@/types/dashboard";

interface StatsCardsProps {
  stats: DashboardStats | null;
  loading?: boolean;
}

interface StatCardProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  loading?: boolean;
}

function StatCard({ label, value, icon, loading }: StatCardProps) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">{label}</p>
        <div className="text-muted-foreground">{icon}</div>
      </div>
      {loading ? (
        <div data-testid="stat-skeleton" className="mt-2 h-8 w-20 animate-pulse rounded bg-muted" />
      ) : (
        <p className="mt-2 text-2xl font-bold text-foreground">{value}</p>
      )}
    </div>
  );
}

export function StatsCards({ stats, loading }: StatsCardsProps) {
  return (
    <div data-testid="stats-cards" className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard
        label="Total Papers"
        value={stats?.total_papers ?? 0}
        icon={<FileText className="size-5" />}
        loading={loading}
      />
      <StatCard
        label="Papers This Week"
        value={stats?.papers_this_week ?? 0}
        icon={<TrendingUp className="size-5" />}
        loading={loading}
      />
      <StatCard
        label="Categories"
        value={stats?.total_categories ?? 0}
        icon={<FolderOpen className="size-5" />}
        loading={loading}
      />
      <StatCard
        label="Most Active"
        value={stats?.most_active_category ?? "N/A"}
        icon={<Star className="size-5" />}
        loading={loading}
      />
    </div>
  );
}
