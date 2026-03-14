"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface SidebarNavItemProps {
  href: string;
  label: string;
  icon: LucideIcon;
  collapsed: boolean;
  shortcut?: string;
}

export function SidebarNavItem({ href, label, icon: Icon, collapsed, shortcut }: SidebarNavItemProps) {
  const pathname = usePathname();
  const isActive = pathname === href || pathname.startsWith(href + "/");

  const link = (
    <Link
      href={href}
      aria-label={label}
      data-active={isActive ? "true" : undefined}
      className={cn(
        "flex items-center gap-3 rounded-lg px-3 py-2.5 text-[15px] font-semibold transition-colors",
        "hover:bg-accent hover:text-accent-foreground",
        isActive
          ? "border-l-3 border-primary bg-primary/10 text-foreground font-bold"
          : "text-muted-foreground border-l-3 border-transparent",
        collapsed && "justify-center px-2"
      )}
    >
      <Icon className="size-5 shrink-0" />
      {!collapsed && (
        <>
          <span className="flex-1">{label}</span>
          {shortcut && (
            <kbd className="ml-auto text-[10px] font-mono text-muted-foreground/60 bg-muted/50 px-1.5 py-0.5 rounded">
              {shortcut}
            </kbd>
          )}
        </>
      )}
    </Link>
  );

  if (collapsed) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{link}</TooltipTrigger>
          <TooltipContent side="right">
            <span>{label}</span>
            {shortcut && (
              <kbd className="ml-2 text-[10px] font-mono text-muted-foreground bg-muted/50 px-1 py-0.5 rounded">
                ⌘{shortcut}
              </kbd>
            )}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return link;
}
