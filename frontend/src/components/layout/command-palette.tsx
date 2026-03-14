"use client";

import { useRouter } from "next/navigation";
import {
  Search,
  MessageSquare,
  Upload,
  FileText,
  Bookmark,
  BarChart3,
  Plus,
  type LucideIcon,
} from "lucide-react";
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandSeparator,
} from "@/components/ui/command";

interface CommandPaletteItem {
  label: string;
  icon: LucideIcon;
  action: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const router = useRouter();

  function navigate(href: string) {
    router.push(href);
    onOpenChange(false);
  }

  const pages: CommandPaletteItem[] = [
    { label: "Search", icon: Search, action: () => navigate("/search") },
    { label: "Chat", icon: MessageSquare, action: () => navigate("/chat") },
    { label: "Upload", icon: Upload, action: () => navigate("/upload") },
    { label: "Papers", icon: FileText, action: () => navigate("/papers") },
    { label: "Collections", icon: Bookmark, action: () => navigate("/collections") },
    { label: "Dashboard", icon: BarChart3, action: () => navigate("/dashboard") },
  ];

  const actions: CommandPaletteItem[] = [
    { label: "New Chat", icon: Plus, action: () => navigate("/chat") },
    { label: "Upload Paper", icon: Upload, action: () => navigate("/upload") },
    { label: "Search Papers", icon: Search, action: () => navigate("/search") },
  ];

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange} title="Command Palette">
      <CommandInput placeholder="Type a command or search..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        <CommandGroup heading="Pages">
          {pages.map((item) => (
            <CommandItem key={item.label} onSelect={item.action}>
              <item.icon className="mr-2 size-4" />
              {item.label}
            </CommandItem>
          ))}
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup heading="Actions">
          {actions.map((item) => (
            <CommandItem key={item.label} onSelect={item.action}>
              <item.icon className="mr-2 size-4" />
              {item.label}
            </CommandItem>
          ))}
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
