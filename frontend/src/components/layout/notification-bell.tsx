"use client";

import { Bell, FileText, MessageSquare, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface NotificationBellProps {
  count?: number;
}

const PLACEHOLDER_NOTIFICATIONS = [
  { icon: FileText, text: "New paper indexed: Attention Is All You Need" },
  { icon: MessageSquare, text: "Chat response ready for your query" },
  { icon: Upload, text: "Paper upload processing complete" },
];

export function NotificationBell({ count = 0 }: NotificationBellProps) {
  const displayCount = count > 99 ? "99+" : String(count);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Notifications" className="relative">
          <Bell className="size-4" />
          {count > 0 && (
            <span
              data-testid="notification-badge"
              className="absolute -right-0.5 -top-0.5 flex size-4 items-center justify-center rounded-full bg-destructive text-[10px] font-bold text-destructive-foreground"
            >
              {displayCount}
            </span>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-72">
        <DropdownMenuLabel>Notifications</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {PLACEHOLDER_NOTIFICATIONS.map((notification) => (
          <DropdownMenuItem key={notification.text}>
            <notification.icon className="mr-2 size-4 shrink-0 text-muted-foreground" />
            <span className="text-sm">{notification.text}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
