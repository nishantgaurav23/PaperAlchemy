"use client";

import { useEffect, useRef, useCallback } from "react";

interface SwipeDelta {
  deltaX: number;
  deltaY: number;
}

interface UseSwipeOptions {
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwiping?: (delta: SwipeDelta) => void;
  threshold?: number;
}

export function useSwipe({
  onSwipeLeft,
  onSwipeRight,
  onSwiping,
  threshold = 50,
}: UseSwipeOptions) {
  const ref = useRef<HTMLElement | null>(null);
  const startX = useRef(0);
  const startY = useRef(0);

  const onSwipeLeftRef = useRef(onSwipeLeft);
  const onSwipeRightRef = useRef(onSwipeRight);
  const onSwipingRef = useRef(onSwiping);

  useEffect(() => {
    onSwipeLeftRef.current = onSwipeLeft;
    onSwipeRightRef.current = onSwipeRight;
    onSwipingRef.current = onSwiping;
  });

  const handleTouchStart = useCallback((e: TouchEvent) => {
    const touch = e.touches[0];
    if (touch) {
      startX.current = touch.clientX;
      startY.current = touch.clientY;
    }
  }, []);

  const handleTouchMove = useCallback((e: TouchEvent) => {
    const touch = e.touches[0];
    if (touch && onSwipingRef.current) {
      const deltaX = touch.clientX - startX.current;
      const deltaY = touch.clientY - startY.current;
      onSwipingRef.current({ deltaX, deltaY });
    }
  }, []);

  const handleTouchEnd = useCallback(
    (e: TouchEvent) => {
      const touch = e.changedTouches[0];
      if (!touch) return;

      const deltaX = touch.clientX - startX.current;
      const deltaY = touch.clientY - startY.current;

      // Only count horizontal swipes (X delta > Y delta)
      if (Math.abs(deltaX) <= Math.abs(deltaY)) return;

      if (deltaX < -threshold && onSwipeLeftRef.current) {
        onSwipeLeftRef.current();
      } else if (deltaX > threshold && onSwipeRightRef.current) {
        onSwipeRightRef.current();
      }
    },
    [threshold]
  );

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    el.addEventListener("touchstart", handleTouchStart, { passive: true });
    el.addEventListener("touchmove", handleTouchMove, { passive: true });
    el.addEventListener("touchend", handleTouchEnd, { passive: true });

    return () => {
      el.removeEventListener("touchstart", handleTouchStart);
      el.removeEventListener("touchmove", handleTouchMove);
      el.removeEventListener("touchend", handleTouchEnd);
    };
  }, [handleTouchStart, handleTouchMove, handleTouchEnd]);

  return { ref };
}
