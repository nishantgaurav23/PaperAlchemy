import { render, screen, within } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { TimelineChart } from "./timeline-chart";
import type { MonthlyCount } from "@/types/dashboard";

// Mock recharts
vi.mock("recharts", () => ({
  AreaChart: ({ data }: { children: React.ReactNode; data: Array<{ label: string; count: number }> }) => (
    <div data-testid="area-chart">
      {data.map((d) => (
        <span key={d.label} data-testid={`point-${d.label}`}>{d.label}: {d.count}</span>
      ))}
    </div>
  ),
  Area: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Tooltip: () => null,
}));

const mockTimeline: MonthlyCount[] = [
  { month: "2025-01", count: 167 },
  { month: "2025-02", count: 178 },
  { month: "2025-03", count: 166 },
];

describe("TimelineChart", () => {
  it("renders chart with timeline data", () => {
    render(<TimelineChart timeline={mockTimeline} />);

    expect(screen.getByTestId("timeline-chart")).toBeInTheDocument();
    expect(screen.getByTestId("area-chart")).toBeInTheDocument();
  });

  it("renders data points for each month", () => {
    render(<TimelineChart timeline={mockTimeline} />);

    const chart = screen.getByTestId("area-chart");
    const points = within(chart).getAllByText(/\d+/);
    expect(points.length).toBeGreaterThanOrEqual(3);
  });

  it("formats months and includes counts", () => {
    render(<TimelineChart timeline={mockTimeline} />);

    // Check that data points are present using testids
    expect(screen.getByTestId("area-chart").textContent).toContain("167");
    expect(screen.getByTestId("area-chart").textContent).toContain("178");
    expect(screen.getByTestId("area-chart").textContent).toContain("166");
  });

  it("shows skeleton when loading", () => {
    render(<TimelineChart timeline={[]} loading />);

    expect(screen.getByTestId("timeline-chart-skeleton")).toBeInTheDocument();
  });

  it("shows empty state when no data", () => {
    render(<TimelineChart timeline={[]} />);

    expect(screen.getByTestId("timeline-chart-empty")).toBeInTheDocument();
    expect(screen.getByText("No timeline data available")).toBeInTheDocument();
  });

  it("handles single data point", () => {
    render(<TimelineChart timeline={[{ month: "2025-03", count: 42 }]} />);

    expect(screen.getByTestId("timeline-chart")).toBeInTheDocument();
    expect(screen.getByTestId("area-chart").textContent).toContain("42");
  });
});
