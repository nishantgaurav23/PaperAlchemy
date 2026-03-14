import * as React from "react";
import { cn } from "@/lib/utils";

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "elevated";
}

function GlassCard({
  className,
  variant = "default",
  ...props
}: GlassCardProps) {
  return (
    <div
      className={cn(
        variant === "elevated" ? "glass-card-elevated" : "glass-card",
        className
      )}
      {...props}
    />
  );
}

export { GlassCard };
export type { GlassCardProps };
