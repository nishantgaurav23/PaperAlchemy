/**
 * S9b.3 — Frontend Infrastructure Dependencies
 * Smoke tests verifying all new packages are importable and functional.
 */
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";

// ---------- FR-1: Markdown Rendering ----------

describe("FR-1: Markdown rendering deps", () => {
  it("imports react-markdown", async () => {
    const mod = await import("react-markdown");
    expect(mod.default).toBeDefined();
  });

  it("imports remark-gfm", async () => {
    const mod = await import("remark-gfm");
    expect(mod.default).toBeDefined();
  });

  it("imports rehype-highlight", async () => {
    const mod = await import("rehype-highlight");
    expect(mod.default).toBeDefined();
  });

  it("renders markdown to HTML", async () => {
    const { default: ReactMarkdown } = await import("react-markdown");
    render(<ReactMarkdown>{"**bold text**"}</ReactMarkdown>);
    expect(screen.getByText("bold text")).toBeInTheDocument();
  });
});

// ---------- FR-2: Animation Library ----------

describe("FR-2: framer-motion", () => {
  it("imports motion components", async () => {
    const mod = await import("framer-motion");
    expect(mod.motion).toBeDefined();
    expect(mod.AnimatePresence).toBeDefined();
  });

  it("renders a motion.div", async () => {
    const { motion } = await import("framer-motion");
    render(<motion.div data-testid="motion-el">hello</motion.div>);
    expect(screen.getByTestId("motion-el")).toHaveTextContent("hello");
  });
});

// ---------- FR-3: Global State Management ----------

describe("FR-3: zustand", () => {
  it("imports zustand create", async () => {
    const mod = await import("zustand");
    expect(mod.create).toBeDefined();
  });

  it("creates and reads a store", async () => {
    const { create } = await import("zustand");

    interface CountStore {
      count: number;
      increment: () => void;
    }

    const useStore = create<CountStore>((set) => ({
      count: 0,
      increment: () => set((s) => ({ count: s.count + 1 })),
    }));

    // Read initial state
    expect(useStore.getState().count).toBe(0);

    // Mutate and read
    useStore.getState().increment();
    expect(useStore.getState().count).toBe(1);
  });
});

// ---------- FR-4: Toast Notifications ----------

describe("FR-4: sonner", () => {
  it("imports toast and Toaster", async () => {
    const mod = await import("sonner");
    expect(mod.toast).toBeDefined();
    expect(mod.Toaster).toBeDefined();
  });

  it("renders Toaster component", async () => {
    const { Toaster } = await import("sonner");
    const { container } = render(<Toaster />);
    expect(container).toBeDefined();
  });
});

// ---------- FR-5: Command Palette ----------

describe("FR-5: cmdk", () => {
  it("imports Command component", async () => {
    const mod = await import("cmdk");
    expect(mod.Command).toBeDefined();
  });

  it("renders Command component", async () => {
    const { Command } = await import("cmdk");
    render(
      <Command data-testid="cmd">
        <Command.Input placeholder="Search..." />
        <Command.List>
          <Command.Item>Item 1</Command.Item>
        </Command.List>
      </Command>
    );
    expect(screen.getByTestId("cmd")).toBeInTheDocument();
  });
});

// ---------- FR-6: Form Validation ----------

describe("FR-6: react-hook-form + zod", () => {
  it("imports useForm", async () => {
    const mod = await import("react-hook-form");
    expect(mod.useForm).toBeDefined();
  });

  it("imports zod", async () => {
    const mod = await import("zod");
    expect(mod.z).toBeDefined();
  });

  it("imports zodResolver", async () => {
    const mod = await import("@hookform/resolvers/zod");
    expect(mod.zodResolver).toBeDefined();
  });

  it("validates with zod schema", async () => {
    const { z } = await import("zod");

    const schema = z.object({
      email: z.string().email(),
      password: z.string().min(8),
    });

    const validResult = schema.safeParse({
      email: "test@example.com",
      password: "password123",
    });
    expect(validResult.success).toBe(true);

    const invalidResult = schema.safeParse({
      email: "not-email",
      password: "short",
    });
    expect(invalidResult.success).toBe(false);
  });
});
