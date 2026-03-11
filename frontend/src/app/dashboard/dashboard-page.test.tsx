import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import DashboardPage from "./page";
import { MOCK_DASHBOARD_DATA } from "@/lib/api/dashboard";

// Mock the dashboard API
vi.mock("@/lib/api/dashboard", () => ({
  getDashboardData: vi.fn(),
  MOCK_DASHBOARD_DATA: {
    stats: {
      total_papers: 1247,
      papers_this_week: 42,
      total_categories: 10,
      most_active_category: "cs.AI",
    },
    categories: [
      { category: "cs.AI", count: 324 },
      { category: "cs.CL", count: 289 },
    ],
    timeline: [
      { month: "2025-01", count: 167 },
      { month: "2025-02", count: 178 },
    ],
    hot_papers: [
      {
        id: "hot-001",
        arxiv_id: "2501.12345",
        title: "Test Paper Title",
        authors: ["Alice Chen"],
        abstract: "Test abstract.",
        categories: ["cs.AI"],
        published_date: "2025-01-15",
      },
    ],
    trending_topics: [
      { topic: "Large Language Models", count: 89 },
    ],
  },
}));

// Mock recharts components
vi.mock("recharts", () => ({
  PieChart: ({ children }: { children: React.ReactNode }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => null,
  Cell: () => null,
  AreaChart: ({ children }: { children: React.ReactNode }) => <div data-testid="area-chart">{children}</div>,
  Area: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Tooltip: () => null,
  Legend: () => null,
}));

// Mock api-client for the mock detection check
vi.mock("@/lib/api-client", () => ({
  apiClient: {
    get: vi.fn().mockRejectedValue(new Error("Not available")),
  },
}));

import { getDashboardData } from "@/lib/api/dashboard";
const mockGetDashboardData = vi.mocked(getDashboardData);

describe("DashboardPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetDashboardData.mockResolvedValue(MOCK_DASHBOARD_DATA);
  });

  it("renders the dashboard page", async () => {
    render(<DashboardPage />);

    expect(screen.getByTestId("dashboard-page")).toBeInTheDocument();
    expect(screen.getByText("Research Trends")).toBeInTheDocument();
  });

  it("renders all dashboard sections", async () => {
    render(<DashboardPage />);

    await waitFor(() => {
      expect(screen.getByText("Category Breakdown")).toBeInTheDocument();
    });

    expect(screen.getByText("Publication Timeline")).toBeInTheDocument();
    expect(screen.getByText("Hot Papers")).toBeInTheDocument();
    expect(screen.getByText("Trending Topics")).toBeInTheDocument();
  });

  it("loads and displays stats", async () => {
    render(<DashboardPage />);

    await waitFor(() => {
      expect(screen.getByText("1247")).toBeInTheDocument();
    });

    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument();
  });

  it("shows loading skeletons initially", () => {
    mockGetDashboardData.mockImplementation(() => new Promise(() => {})); // never resolves
    render(<DashboardPage />);

    expect(screen.getAllByTestId("stat-skeleton")).toHaveLength(4);
  });

  it("shows mock data banner when using sample data", async () => {
    render(<DashboardPage />);

    await waitFor(() => {
      expect(screen.getByTestId("mock-data-banner")).toBeInTheDocument();
    });

    expect(screen.getByText(/Using sample data/)).toBeInTheDocument();
  });

  it("renders hot papers from data", async () => {
    render(<DashboardPage />);

    await waitFor(() => {
      expect(screen.getByText("Test Paper Title")).toBeInTheDocument();
    });
  });

  it("renders trending topics from data", async () => {
    render(<DashboardPage />);

    await waitFor(() => {
      expect(screen.getByText("Large Language Models")).toBeInTheDocument();
    });
  });
});
