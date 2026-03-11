import {
  Search,
  MessageSquare,
  Upload,
  FileText,
  Bookmark,
  BarChart3,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  label: string;
  href: string;
  icon: LucideIcon;
}

export const NAV_ITEMS: NavItem[] = [
  { label: "Search", href: "/search", icon: Search },
  { label: "Chat", href: "/chat", icon: MessageSquare },
  { label: "Upload", href: "/upload", icon: Upload },
  { label: "Papers", href: "/papers", icon: FileText },
  { label: "Collections", href: "/collections", icon: Bookmark },
  { label: "Dashboard", href: "/dashboard", icon: BarChart3 },
];
