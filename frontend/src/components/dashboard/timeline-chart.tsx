"use client";

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip } from "recharts";
import type { MonthlyCount } from "@/types/dashboard";

interface TimelineChartProps {
  timeline: MonthlyCount[];
  loading?: boolean;
}

function formatMonth(month: string): string {
  const [year, m] = month.split("-");
  const date = new Date(Number(year), Number(m) - 1);
  return date.toLocaleDateString("en-US", { month: "short", year: "2-digit" });
}

export function TimelineChart({ timeline, loading }: TimelineChartProps) {
  if (loading) {
    return (
      <div data-testid="timeline-chart-skeleton" className="flex h-[300px] items-center justify-center">
        <div className="h-full w-full animate-pulse rounded bg-muted" />
      </div>
    );
  }

  if (timeline.length === 0) {
    return (
      <div data-testid="timeline-chart-empty" className="flex h-[300px] items-center justify-center text-muted-foreground">
        No timeline data available
      </div>
    );
  }

  const data = timeline.map((t) => ({
    ...t,
    label: formatMonth(t.month),
  }));

  return (
    <div data-testid="timeline-chart" className="h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(221, 83%, 53%)" stopOpacity={0.3} />
              <stop offset="95%" stopColor="hsl(221, 83%, 53%)" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 12 }}
            className="fill-muted-foreground"
          />
          <YAxis
            tick={{ fontSize: 12 }}
            className="fill-muted-foreground"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              borderColor: "hsl(var(--border))",
              borderRadius: "0.5rem",
            }}
            formatter={(value: number) => [value, "Papers"]}
          />
          <Area
            type="monotone"
            dataKey="count"
            stroke="hsl(221, 83%, 53%)"
            fillOpacity={1}
            fill="url(#colorCount)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
