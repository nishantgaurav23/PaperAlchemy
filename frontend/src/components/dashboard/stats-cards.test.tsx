import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { StatsCards } from "./stats-cards";
import type { DashboardStats } from "@/types/dashboard";

const mockStats: DashboardStats = {
  total_papers: 1247,
  papers_this_week: 42,
  total_categories: 10,
  most_active_category: "cs.AI",
};

describe("StatsCards", () => {
  it("renders all four stat cards", () => {
    render(<StatsCards stats={mockStats} />);

    expect(screen.getByText("Total Papers")).toBeInTheDocument();
    expect(screen.getByText("Papers This Week")).toBeInTheDocument();
    expect(screen.getByText("Categories")).toBeInTheDocument();
    expect(screen.getByText("Most Active")).toBeInTheDocument();
  });

  it("displays stat values", () => {
    render(<StatsCards stats={mockStats} />);

    expect(screen.getByText("1247")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getByText("cs.AI")).toBeInTheDocument();
  });

  it("shows skeletons when loading", () => {
    render(<StatsCards stats={null} loading />);

    const skeletons = screen.getAllByTestId("stat-skeleton");
    expect(skeletons).toHaveLength(4);
  });

  it("shows zero values when stats is null and not loading", () => {
    render(<StatsCards stats={null} />);

    expect(screen.getAllByText("0")).toHaveLength(3);
    expect(screen.getByText("N/A")).toBeInTheDocument();
  });

  it("has the stats-cards test id", () => {
    render(<StatsCards stats={mockStats} />);
    expect(screen.getByTestId("stats-cards")).toBeInTheDocument();
  });
});
