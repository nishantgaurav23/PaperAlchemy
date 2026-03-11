import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { CategoryChart } from "./category-chart";
import type { CategoryCount } from "@/types/dashboard";

// Mock recharts - jsdom doesn't support SVG rendering
vi.mock("recharts", () => ({
  PieChart: ({ children }: { children: React.ReactNode }) => <div data-testid="pie-chart">{children}</div>,
  Pie: ({ data }: { data: CategoryCount[] }) => (
    <div data-testid="pie">
      {data.map((d) => (
        <span key={d.category}>{d.category}: {d.count}</span>
      ))}
    </div>
  ),
  Cell: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Tooltip: () => null,
  Legend: () => null,
}));

const mockCategories: CategoryCount[] = [
  { category: "cs.AI", count: 324 },
  { category: "cs.CL", count: 289 },
  { category: "cs.LG", count: 256 },
];

describe("CategoryChart", () => {
  it("renders chart with category data", () => {
    render(<CategoryChart categories={mockCategories} />);

    expect(screen.getByTestId("category-chart")).toBeInTheDocument();
    expect(screen.getByTestId("pie-chart")).toBeInTheDocument();
    expect(screen.getByText(/cs\.AI: 324/)).toBeInTheDocument();
    expect(screen.getByText(/cs\.CL: 289/)).toBeInTheDocument();
  });

  it("shows skeleton when loading", () => {
    render(<CategoryChart categories={[]} loading />);

    expect(screen.getByTestId("category-chart-skeleton")).toBeInTheDocument();
  });

  it("shows empty state when no categories", () => {
    render(<CategoryChart categories={[]} />);

    expect(screen.getByTestId("category-chart-empty")).toBeInTheDocument();
    expect(screen.getByText("No category data available")).toBeInTheDocument();
  });

  it("groups excess categories into Other", () => {
    const many: CategoryCount[] = Array.from({ length: 12 }, (_, i) => ({
      category: `cat.${i}`,
      count: 100 - i * 5,
    }));

    render(<CategoryChart categories={many} />);

    // First 8 should be shown directly
    expect(screen.getByText(/cat\.0/)).toBeInTheDocument();
    expect(screen.getByText(/cat\.7/)).toBeInTheDocument();
    // 9th+ should be grouped as "Other"
    expect(screen.getByText(/Other/)).toBeInTheDocument();
    expect(screen.queryByText(/cat\.8/)).not.toBeInTheDocument();
  });
});
