import {
  Search,
  MessageSquare,
  FileText,
  Bookmark,
  BarChart3,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  label: string;
  href: string;
  icon: LucideIcon;
  shortcut?: string;
}

export const NAV_ITEMS: NavItem[] = [
  { label: "Search", href: "/search", icon: Search, shortcut: "1" },
  { label: "Chat", href: "/chat", icon: MessageSquare, shortcut: "2" },
  { label: "Papers", href: "/papers", icon: FileText, shortcut: "3" },
  { label: "Collections", href: "/collections", icon: Bookmark, shortcut: "4" },
  { label: "Dashboard", href: "/dashboard", icon: BarChart3, shortcut: "5" },
];
