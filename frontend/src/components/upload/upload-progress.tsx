"use client";

import { FileText, CheckCircle2, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { UploadStatus } from "@/types/upload";

interface UploadProgressProps {
  status: UploadStatus;
  fileName: string;
}

const STATUS_CONFIG: Record<
  Exclude<UploadStatus, "idle" | "error">,
  { label: string; icon: typeof Loader2; color: string }
> = {
  uploading: { label: "Uploading...", icon: Loader2, color: "text-primary" },
  processing: { label: "Processing & analyzing...", icon: Loader2, color: "text-amber-500" },
  complete: { label: "Analysis complete!", icon: CheckCircle2, color: "text-green-500" },
};

export function UploadProgress({ status, fileName }: UploadProgressProps) {
  if (status === "idle" || status === "error") return null;

  const config = STATUS_CONFIG[status];
  const Icon = config.icon;
  const isAnimating = status === "uploading" || status === "processing";

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border bg-card p-6">
      <div className="flex items-center gap-3">
        <FileText className="size-5 text-muted-foreground" />
        <span className="text-sm font-medium">{fileName}</span>
      </div>

      <div className="flex items-center gap-2">
        <Icon className={cn("size-4", config.color, isAnimating && "animate-spin")} />
        <span className={cn("text-sm font-medium", config.color)}>{config.label}</span>
      </div>

      <div
        role="progressbar"
        aria-valuenow={status === "complete" ? 100 : status === "processing" ? 80 : 40}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-2 w-full overflow-hidden rounded-full bg-muted"
      >
        <div
          data-testid="progress-fill"
          className={cn(
            "h-full rounded-full transition-all duration-500",
            status === "uploading" && "w-2/5 animate-pulse bg-primary",
            status === "processing" && "w-4/5 animate-pulse bg-amber-500",
            status === "complete" && "w-full bg-green-500",
          )}
        />
      </div>
    </div>
  );
}
