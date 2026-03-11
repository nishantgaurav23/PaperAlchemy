import { Skeleton } from "@/components/ui/skeleton";

export function PaperDetailSkeleton() {
  return (
    <div data-testid="paper-detail-skeleton" className="flex flex-col gap-6">
      {/* Title */}
      <Skeleton className="h-8 w-3/4" />
      {/* Authors */}
      <Skeleton className="h-4 w-1/2" />
      {/* Date + badges */}
      <div className="flex gap-2">
        <Skeleton className="h-5 w-24" />
        <Skeleton className="h-5 w-16" />
        <Skeleton className="h-5 w-16" />
      </div>
      {/* Links */}
      <div className="flex gap-2">
        <Skeleton className="h-8 w-16" />
        <Skeleton className="h-8 w-16" />
        <Skeleton className="h-8 w-28" />
      </div>
      {/* Abstract */}
      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-2/3" />
      </div>
      {/* Sections */}
      <div className="flex flex-col gap-2">
        <Skeleton className="h-6 w-24" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    </div>
  );
}
