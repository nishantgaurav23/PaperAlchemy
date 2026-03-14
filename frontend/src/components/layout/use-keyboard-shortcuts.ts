"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { NAV_ITEMS } from "./nav-items";

interface UseKeyboardShortcutsOptions {
  onCommandK?: () => void;
}

export function useKeyboardShortcuts({ onCommandK }: UseKeyboardShortcutsOptions = {}) {
  const router = useRouter();

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const modifier = e.metaKey || e.ctrlKey;
      if (!modifier) return;

      // Skip when focused on input/textarea/contenteditable
      const target = e.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }

      // Cmd+K → command palette
      if (e.key === "k") {
        e.preventDefault();
        onCommandK?.();
        return;
      }

      // Cmd+1-6 → nav shortcuts
      const index = parseInt(e.key, 10);
      if (index >= 1 && index <= NAV_ITEMS.length) {
        e.preventDefault();
        router.push(NAV_ITEMS[index - 1].href);
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [router, onCommandK]);
}
