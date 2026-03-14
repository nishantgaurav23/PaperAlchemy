import { cn } from "@/lib/utils";

interface ProgressIndicatorProps {
  value?: number;
  variant?: "default" | "error" | "success";
  label?: string;
  showPercentage?: boolean;
  className?: string;
}

const variantClasses = {
  default: "bg-primary",
  error: "bg-destructive",
  success: "bg-success",
};

export function ProgressIndicator({
  value,
  variant = "default",
  label,
  showPercentage,
  className,
}: ProgressIndicatorProps) {
  const isDeterminate = value !== undefined;

  return (
    <div className="flex flex-col gap-1.5">
      {(label || showPercentage) && (
        <div className="flex items-center justify-between text-sm">
          {label && (
            <span className="text-muted-foreground">{label}</span>
          )}
          {showPercentage && isDeterminate && (
            <span className="text-muted-foreground">{value}%</span>
          )}
        </div>
      )}
      <div
        role="progressbar"
        aria-valuenow={isDeterminate ? value : undefined}
        aria-valuemin={0}
        aria-valuemax={100}
        className={cn(
          "h-2 w-full overflow-hidden rounded-full bg-muted",
          className
        )}
      >
        <div
          data-testid="progress-fill"
          className={cn(
            "h-full rounded-full transition-all duration-500",
            variantClasses[variant],
            !isDeterminate && "animate-progress-indeterminate w-1/3"
          )}
          style={isDeterminate ? { width: `${value}%` } : undefined}
        />
      </div>
    </div>
  );
}
