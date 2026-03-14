"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ArrowDown } from "lucide-react";

interface ScrollToBottomProps {
  onClick: () => void;
  visible: boolean;
}

export function ScrollToBottom({ onClick, visible }: ScrollToBottomProps) {
  return (
    <AnimatePresence>
      {visible && (
        <motion.button
          initial={{ opacity: 0, scale: 0.8, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.8, y: 10 }}
          transition={{ duration: 0.15 }}
          onClick={onClick}
          aria-label="Scroll to bottom"
          className="absolute bottom-20 right-4 z-10 inline-flex size-9 items-center justify-center rounded-full border border-border bg-background shadow-md transition-colors hover:bg-accent"
        >
          <ArrowDown className="size-4" />
        </motion.button>
      )}
    </AnimatePresence>
  );
}
