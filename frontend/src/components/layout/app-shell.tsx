"use client";

import { useState } from "react";
import { Sidebar } from "./sidebar";
import { Header } from "./header";
import { MobileNav } from "./mobile-nav";
import { BottomNav } from "./bottom-nav";
import { CommandPalette } from "./command-palette";
import { useKeyboardShortcuts } from "./use-keyboard-shortcuts";
import { ToastProvider } from "@/components/animations/toast-provider";

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);

  useKeyboardShortcuts({
    onCommandK: () => setCommandPaletteOpen(true),
  });

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <MobileNav open={mobileNavOpen} onClose={() => setMobileNavOpen(false)} />
      <CommandPalette open={commandPaletteOpen} onOpenChange={setCommandPaletteOpen} />

      <div className="flex flex-1 flex-col overflow-hidden">
        <Header
          showMobileMenu
          onMobileMenuToggle={() => setMobileNavOpen(true)}
        />
        <main className="flex-1 overflow-y-auto p-4 pb-20 md:p-6 md:pb-6">
          {children}
        </main>
      </div>

      <BottomNav />
      <ToastProvider />
    </div>
  );
}
