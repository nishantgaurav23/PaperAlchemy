"use client";

import { useState } from "react";
import { ChevronsLeft, ChevronsRight, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SidebarNavItem } from "./sidebar-nav-item";
import { NAV_ITEMS } from "./nav-items";
import { cn } from "@/lib/utils";

const STORAGE_KEY = "sidebar-collapsed";

function getInitialCollapsed(): boolean {
  if (typeof window === "undefined") return false;
  return localStorage.getItem(STORAGE_KEY) === "true";
}

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(getInitialCollapsed);

  function toggleCollapse() {
    const next = !collapsed;
    setCollapsed(next);
    localStorage.setItem(STORAGE_KEY, String(next));
  }

  return (
    <aside
      className={cn(
        "hidden md:flex flex-col border-r border-border bg-sidebar text-sidebar-foreground transition-[width] duration-200",
        collapsed ? "w-16" : "w-56"
      )}
    >
      <div className={cn("flex h-14 items-center border-b border-border px-4", collapsed && "justify-center px-2")}>
        <FlaskConical className="size-5 shrink-0 text-primary" />
        {!collapsed && <span className="ml-2 text-lg font-semibold tracking-tight">PaperAlchemy</span>}
      </div>

      <nav aria-label="Main navigation" className="flex-1 space-y-1 p-2">
        {NAV_ITEMS.map((item) => (
          <SidebarNavItem
            key={item.href}
            href={item.href}
            label={item.label}
            icon={item.icon}
            collapsed={collapsed}
          />
        ))}
      </nav>

      <div className="border-t border-border p-2">
        <Button
          variant="ghost"
          size={collapsed ? "icon" : "default"}
          onClick={toggleCollapse}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className={cn("w-full", !collapsed && "justify-start")}
        >
          {collapsed ? <ChevronsRight className="size-4" /> : <ChevronsLeft className="size-4" />}
          {!collapsed && <span className="ml-2">Collapse</span>}
        </Button>
      </div>
    </aside>
  );
}
