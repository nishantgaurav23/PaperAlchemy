import { cn } from "@/lib/utils";

type SkeletonCardProps = React.ComponentProps<"div">;

export function SkeletonCard({ className, ...props }: SkeletonCardProps) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-xl bg-muted",
        "h-48 w-full",
        className
      )}
      {...props}
    />
  );
}

interface SkeletonTextProps extends React.ComponentProps<"div"> {
  lines?: number;
}

export function SkeletonText({ lines = 1, className, ...props }: SkeletonTextProps) {
  return (
    <div className={cn("space-y-2", className)} {...props}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          data-slot="skeleton-line"
          className={cn(
            "animate-pulse rounded-md bg-muted h-4",
            i === lines - 1 && lines > 1 ? "w-3/4" : "w-full"
          )}
        />
      ))}
    </div>
  );
}

type SkeletonChartProps = React.ComponentProps<"div">;

export function SkeletonChart({ className, ...props }: SkeletonChartProps) {
  return (
    <div
      className={cn("shimmer rounded-xl aspect-video w-full", className)}
      {...props}
    />
  );
}

interface SkeletonListProps extends React.ComponentProps<"div"> {
  count?: number;
}

export function SkeletonList({ count = 3, className, ...props }: SkeletonListProps) {
  return (
    <div className={cn("space-y-3", className)} {...props}>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          data-slot="skeleton-item"
          className="animate-pulse rounded-lg bg-muted h-16 w-full"
        />
      ))}
    </div>
  );
}
