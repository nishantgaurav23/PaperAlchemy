"use client";

import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { Breadcrumbs } from "./breadcrumbs";

interface HeaderProps {
  onMobileMenuToggle?: () => void;
  showMobileMenu?: boolean;
}

export function Header({ onMobileMenuToggle, showMobileMenu }: HeaderProps) {
  return (
    <header className="flex h-14 items-center gap-4 border-b border-border bg-background px-4">
      {showMobileMenu && (
        <Button
          variant="ghost"
          size="icon"
          onClick={onMobileMenuToggle}
          aria-label="Open menu"
          className="md:hidden"
        >
          <Menu className="size-5" />
        </Button>
      )}

      <div className="flex-1">
        <Breadcrumbs />
      </div>

      <ThemeToggle />
    </header>
  );
}
