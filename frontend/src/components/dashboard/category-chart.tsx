"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";
import type { CategoryCount } from "@/types/dashboard";

interface CategoryChartProps {
  categories: CategoryCount[];
  loading?: boolean;
}

const COLORS = [
  "hsl(221, 83%, 53%)", // blue
  "hsl(262, 83%, 58%)", // purple
  "hsl(142, 71%, 45%)", // green
  "hsl(38, 92%, 50%)",  // amber
  "hsl(0, 84%, 60%)",   // red
  "hsl(190, 90%, 50%)", // cyan
  "hsl(330, 81%, 60%)", // pink
  "hsl(25, 95%, 53%)",  // orange
  "hsl(200, 18%, 46%)", // slate (Other)
];

function prepareData(categories: CategoryCount[]) {
  if (categories.length <= 8) return categories;
  const top = categories.slice(0, 8);
  const otherCount = categories.slice(8).reduce((sum, c) => sum + c.count, 0);
  return [...top, { category: "Other", count: otherCount }];
}

export function CategoryChart({ categories, loading }: CategoryChartProps) {
  if (loading) {
    return (
      <div data-testid="category-chart-skeleton" className="flex h-[300px] items-center justify-center">
        <div className="h-48 w-48 animate-pulse rounded-full bg-muted" />
      </div>
    );
  }

  if (categories.length === 0) {
    return (
      <div data-testid="category-chart-empty" className="flex h-[300px] items-center justify-center text-muted-foreground">
        No category data available
      </div>
    );
  }

  const data = prepareData(categories);

  return (
    <div data-testid="category-chart" className="h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="count"
            nameKey="category"
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={100}
            paddingAngle={2}
          >
            {data.map((entry, index) => (
              <Cell key={entry.category} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              borderColor: "hsl(var(--border))",
              borderRadius: "0.5rem",
            }}
            formatter={(value) => [String(value), "Papers"]}
          />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
