"use client";

import { AnimatedCounter } from "@/components/animations/animated-counter";

interface Stat {
  label: string;
  value: number;
  suffix: string;
}

const stats: Stat[] = [
  { label: "Papers Indexed", value: 10000, suffix: "+" },
  { label: "Questions Answered", value: 50000, suffix: "+" },
  { label: "Citations Generated", value: 150000, suffix: "+" },
];

function StatItem({ stat }: { stat: Stat }) {
  return (
    <div data-testid="stat-item" className="flex flex-col items-center gap-2">
      <span
        data-testid="stat-number"
        className="text-4xl font-bold tracking-tight text-primary"
      >
        <AnimatedCounter value={stat.value} abbreviate suffix={stat.suffix} />
      </span>
      <span
        data-testid="stat-label"
        className="text-sm text-muted-foreground"
      >
        {stat.label}
      </span>
    </div>
  );
}

export function StatsCounter() {
  return (
    <section className="border-y bg-muted/50 px-4 py-20">
      <div className="mx-auto flex max-w-4xl flex-col items-center gap-12 sm:flex-row sm:justify-around">
        {stats.map((stat) => (
          <StatItem key={stat.label} stat={stat} />
        ))}
      </div>
    </section>
  );
}
