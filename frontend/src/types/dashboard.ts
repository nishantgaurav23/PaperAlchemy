import type { Paper } from "./paper";

export interface DashboardStats {
  total_papers: number;
  papers_this_week: number;
  total_categories: number;
  most_active_category: string;
}

export interface CategoryCount {
  category: string;
  count: number;
}

export interface MonthlyCount {
  month: string; // "2024-01", "2024-02", etc.
  count: number;
}

export interface TrendingTopic {
  topic: string;
  count: number;
}

export interface DashboardData {
  stats: DashboardStats;
  categories: CategoryCount[];
  timeline: MonthlyCount[];
  hot_papers: Paper[];
  trending_topics: TrendingTopic[];
}
