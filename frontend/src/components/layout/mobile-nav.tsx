"use client";

import { X, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SidebarNavItem } from "./sidebar-nav-item";
import { NAV_ITEMS } from "./nav-items";

interface MobileNavProps {
  open: boolean;
  onClose: () => void;
}

export function MobileNav({ open, onClose }: MobileNavProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 md:hidden">
      {/* Backdrop */}
      <div
        data-testid="mobile-nav-backdrop"
        className="fixed inset-0 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        role="dialog"
        aria-label="Navigation menu"
        className="fixed inset-y-0 left-0 w-64 border-r border-border bg-sidebar text-sidebar-foreground shadow-lg"
      >
        <div className="flex h-14 items-center justify-between border-b border-border px-4">
          <div className="flex items-center gap-2">
            <FlaskConical className="size-5 text-primary" />
            <span className="text-lg font-semibold tracking-tight">PaperAlchemy</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="Close menu"
          >
            <X className="size-4" />
          </Button>
        </div>

        <nav aria-label="Mobile navigation" className="space-y-1 p-2">
          {NAV_ITEMS.map((item) => (
            <SidebarNavItem
              key={item.href}
              href={item.href}
              label={item.label}
              icon={item.icon}
              collapsed={false}
            />
          ))}
        </nav>
      </div>
    </div>
  );
}
