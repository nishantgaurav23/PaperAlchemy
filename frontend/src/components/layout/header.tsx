"use client";

import Link from "next/link";
import { LogIn, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { useAuthStore } from "@/lib/auth/store";
import { Breadcrumbs } from "./breadcrumbs";
import { NotificationBell } from "./notification-bell";

interface HeaderProps {
  onMobileMenuToggle?: () => void;
  showMobileMenu?: boolean;
}

export function Header({ onMobileMenuToggle, showMobileMenu }: HeaderProps) {
  const { isAuthenticated, user, logout } = useAuthStore();

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

      <div className="flex items-center gap-2">
        <NotificationBell count={3} />
        {isAuthenticated ? (
          <>
            <span className="text-sm text-muted-foreground">{user?.name}</span>
            <Button variant="ghost" size="sm" onClick={logout}>
              Sign out
            </Button>
          </>
        ) : (
          <Button variant="ghost" size="sm" asChild>
            <Link href="/login">
              <LogIn className="mr-1 size-4" />
              Sign in
            </Link>
          </Button>
        )}
        <ThemeToggle />
      </div>
    </header>
  );
}
