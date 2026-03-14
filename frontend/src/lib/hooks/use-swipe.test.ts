import { renderHook, act } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { useSwipe } from "./use-swipe";

function createTouchEvent(type: string, clientX: number, clientY: number) {
  return new TouchEvent(type, {
    touches: type === "touchend" ? [] : [{ clientX, clientY } as Touch],
    changedTouches: [{ clientX, clientY } as Touch],
    bubbles: true,
  });
}

describe("useSwipe", () => {
  it("returns a ref to attach to the target element", () => {
    const { result } = renderHook(() => useSwipe({}));
    expect(result.current.ref).toBeDefined();
  });

  it("fires onSwipeLeft when swiping left past threshold", () => {
    const onSwipeLeft = vi.fn();
    const div = document.createElement("div");

    renderHook(() => {
      const hook = useSwipe({ onSwipeLeft, threshold: 50 });
      (hook.ref as React.MutableRefObject<HTMLElement | null>).current = div;
      return hook;
    });

    act(() => {
      div.dispatchEvent(createTouchEvent("touchstart", 200, 100));
      div.dispatchEvent(createTouchEvent("touchmove", 100, 100));
      div.dispatchEvent(createTouchEvent("touchend", 100, 100));
    });

    expect(onSwipeLeft).toHaveBeenCalledTimes(1);
  });

  it("fires onSwipeRight when swiping right past threshold", () => {
    const onSwipeRight = vi.fn();
    const div = document.createElement("div");

    renderHook(() => {
      const hook = useSwipe({ onSwipeRight, threshold: 50 });
      (hook.ref as React.MutableRefObject<HTMLElement | null>).current = div;
      return hook;
    });

    act(() => {
      div.dispatchEvent(createTouchEvent("touchstart", 100, 100));
      div.dispatchEvent(createTouchEvent("touchmove", 200, 100));
      div.dispatchEvent(createTouchEvent("touchend", 200, 100));
    });

    expect(onSwipeRight).toHaveBeenCalledTimes(1);
  });

  it("does not fire callback when swipe is below threshold", () => {
    const onSwipeLeft = vi.fn();
    const onSwipeRight = vi.fn();
    const div = document.createElement("div");

    renderHook(() => {
      const hook = useSwipe({ onSwipeLeft, onSwipeRight, threshold: 50 });
      (hook.ref as React.MutableRefObject<HTMLElement | null>).current = div;
      return hook;
    });

    act(() => {
      div.dispatchEvent(createTouchEvent("touchstart", 100, 100));
      div.dispatchEvent(createTouchEvent("touchmove", 130, 100));
      div.dispatchEvent(createTouchEvent("touchend", 130, 100));
    });

    expect(onSwipeLeft).not.toHaveBeenCalled();
    expect(onSwipeRight).not.toHaveBeenCalled();
  });

  it("ignores vertical swipes (Y delta > X delta)", () => {
    const onSwipeLeft = vi.fn();
    const div = document.createElement("div");

    renderHook(() => {
      const hook = useSwipe({ onSwipeLeft, threshold: 50 });
      (hook.ref as React.MutableRefObject<HTMLElement | null>).current = div;
      return hook;
    });

    act(() => {
      div.dispatchEvent(createTouchEvent("touchstart", 200, 100));
      div.dispatchEvent(createTouchEvent("touchmove", 150, 300));
      div.dispatchEvent(createTouchEvent("touchend", 150, 300));
    });

    expect(onSwipeLeft).not.toHaveBeenCalled();
  });

  it("uses default threshold of 50px", () => {
    const onSwipeLeft = vi.fn();
    const div = document.createElement("div");

    renderHook(() => {
      const hook = useSwipe({ onSwipeLeft });
      (hook.ref as React.MutableRefObject<HTMLElement | null>).current = div;
      return hook;
    });

    // Swipe 49px — should not fire
    act(() => {
      div.dispatchEvent(createTouchEvent("touchstart", 100, 100));
      div.dispatchEvent(createTouchEvent("touchmove", 51, 100));
      div.dispatchEvent(createTouchEvent("touchend", 51, 100));
    });

    expect(onSwipeLeft).not.toHaveBeenCalled();

    // Swipe 51px — should fire
    act(() => {
      div.dispatchEvent(createTouchEvent("touchstart", 100, 100));
      div.dispatchEvent(createTouchEvent("touchmove", 49, 100));
      div.dispatchEvent(createTouchEvent("touchend", 49, 100));
    });

    expect(onSwipeLeft).toHaveBeenCalledTimes(1);
  });

  it("provides swipe delta via onSwiping callback", () => {
    const onSwiping = vi.fn();
    const div = document.createElement("div");

    renderHook(() => {
      const hook = useSwipe({ onSwiping });
      (hook.ref as React.MutableRefObject<HTMLElement | null>).current = div;
      return hook;
    });

    act(() => {
      div.dispatchEvent(createTouchEvent("touchstart", 100, 100));
      div.dispatchEvent(createTouchEvent("touchmove", 150, 100));
    });

    expect(onSwiping).toHaveBeenCalledWith({ deltaX: 50, deltaY: 0 });
  });
});
