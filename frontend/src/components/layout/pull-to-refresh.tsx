"use client";

import { useState, useRef, useCallback, type ReactNode } from "react";
import { Loader2 } from "lucide-react";

interface PullToRefreshProps {
  onRefresh: () => Promise<void>;
  threshold?: number;
  children: ReactNode;
}

export function PullToRefresh({
  onRefresh,
  threshold = 60,
  children,
}: PullToRefreshProps) {
  const [pullDistance, setPullDistance] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const startY = useRef(0);
  const pulling = useRef(false);
  const pullDistanceRef = useRef(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleTouchStart = useCallback(
    (e: React.TouchEvent) => {
      if (isRefreshing) return;
      const container = containerRef.current;
      if (!container || container.scrollTop > 0) return;

      const touch = e.touches[0];
      if (touch) {
        startY.current = touch.clientY;
        pulling.current = true;
      }
    },
    [isRefreshing]
  );

  const handleTouchMove = useCallback(
    (e: React.TouchEvent) => {
      if (!pulling.current || isRefreshing) return;
      const touch = e.touches[0];
      if (!touch) return;

      const delta = touch.clientY - startY.current;
      if (delta > 0) {
        const dampened = Math.min(delta * 0.5, 120);
        pullDistanceRef.current = dampened;
        setPullDistance(dampened);
      }
    },
    [isRefreshing]
  );

  const handleTouchEnd = useCallback(async () => {
    if (!pulling.current) return;
    pulling.current = false;

    const currentPull = pullDistanceRef.current;
    if (currentPull >= threshold) {
      setIsRefreshing(true);
      setPullDistance(threshold);
      try {
        await onRefresh();
      } finally {
        setIsRefreshing(false);
        setPullDistance(0);
        pullDistanceRef.current = 0;
      }
    } else {
      setPullDistance(0);
      pullDistanceRef.current = 0;
    }
  }, [threshold, onRefresh]);

  return (
    <div
      ref={containerRef}
      data-testid="pull-to-refresh-container"
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
      className="relative"
    >
      {/* Pull indicator — mobile only */}
      {(pullDistance > 0 || isRefreshing) && (
        <div
          data-testid={isRefreshing ? "pull-refreshing" : "pull-indicator"}
          className="flex items-center justify-center overflow-hidden transition-[height] md:hidden"
          style={{ height: isRefreshing ? threshold : pullDistance }}
        >
          {isRefreshing ? (
            <Loader2 className="size-5 animate-spin text-primary" />
          ) : (
            <div className="flex flex-col items-center gap-1">
              <Loader2
                className="size-4 text-muted-foreground transition-transform"
                style={{
                  transform: `rotate(${(pullDistance / threshold) * 360}deg)`,
                  opacity: Math.min(pullDistance / threshold, 1),
                }}
              />
              <span className="text-xs text-muted-foreground">
                {pullDistance >= threshold ? "Release to refresh" : "Pull to refresh"}
              </span>
            </div>
          )}
        </div>
      )}

      {children}
    </div>
  );
}
