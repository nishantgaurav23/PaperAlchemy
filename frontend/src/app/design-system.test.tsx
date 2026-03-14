import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import fs from "fs";
import path from "path";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { GlassCard } from "@/components/ui/glass-card";

// Read globals.css once for all CSS-based tests
const cssPath = path.resolve(__dirname, "globals.css");
const css = fs.readFileSync(cssPath, "utf-8");

// ─── FR-1: Color Palette ────────────────────────────────────────────────────

describe("FR-1: Color Palette (Indigo/Violet)", () => {
  it("defines indigo/violet primary tokens in :root (not grayscale)", () => {
    const rootBlock = css.match(/:root\s*\{([\s\S]*?)\n\}/)?.[1] ?? "";
    expect(rootBlock).toContain("--primary:");
    expect(rootBlock).toContain("--primary-foreground:");
    // Primary should have chroma > 0 (not grayscale)
    const primaryMatch = rootBlock.match(/--primary:\s*oklch\(([^)]+)\)/);
    expect(primaryMatch).toBeTruthy();
    const parts = primaryMatch![1].split(/\s+/);
    expect(parseFloat(parts[1])).toBeGreaterThan(0);
  });

  it("defines success and warning semantic tokens", () => {
    expect(css).toContain("--success:");
    expect(css).toContain("--warning:");
  });

  it("dark mode has indigo/violet primary (not grayscale)", () => {
    const darkBlock = css.match(/\.dark\s*\{([\s\S]*?)\n\}/)?.[1] ?? "";
    expect(darkBlock).toContain("--primary:");
    const primaryMatch = darkBlock.match(/--primary:\s*oklch\(([^)]+)\)/);
    expect(primaryMatch).toBeTruthy();
    const parts = primaryMatch![1].split(/\s+/);
    expect(parseFloat(parts[1])).toBeGreaterThan(0);
  });

  it("sidebar tokens use primary-derived colors", () => {
    const rootBlock = css.match(/:root\s*\{([\s\S]*?)\n\}/)?.[1] ?? "";
    expect(rootBlock).toContain("--sidebar-primary:");
    const match = rootBlock.match(/--sidebar-primary:\s*oklch\(([^)]+)\)/);
    expect(match).toBeTruthy();
    const parts = match![1].split(/\s+/);
    expect(parseFloat(parts[1])).toBeGreaterThan(0);
  });

  it("maps success and warning to Tailwind via @theme", () => {
    const themeBlock = css.match(/@theme inline\s*\{([\s\S]*?)\n\}/)?.[1] ?? "";
    expect(themeBlock).toContain("--color-success:");
    expect(themeBlock).toContain("--color-warning:");
  });
});

// ─── FR-2: Glassmorphism ────────────────────────────────────────────────────

describe("FR-2: Glassmorphism Card Styles", () => {
  it("GlassCard renders with glass-card class", () => {
    render(<GlassCard data-testid="glass">Content</GlassCard>);
    const card = screen.getByTestId("glass");
    expect(card).toBeInTheDocument();
    expect(card.className).toContain("glass-card");
  });

  it("GlassCard elevated variant adds elevated class", () => {
    render(
      <GlassCard variant="elevated" data-testid="glass-elevated">
        Content
      </GlassCard>
    );
    const card = screen.getByTestId("glass-elevated");
    expect(card.className).toContain("glass-card-elevated");
  });

  it("globals.css defines .glass-card with backdrop-filter", () => {
    expect(css).toContain(".glass-card");
    expect(css).toContain("backdrop-filter");
  });

  it("globals.css defines .glass-card-elevated with shadow", () => {
    expect(css).toContain(".glass-card-elevated");
    expect(css).toContain("box-shadow");
  });
});

// ─── FR-4: Typography Scale ─────────────────────────────────────────────────

describe("FR-4: Typography Scale", () => {
  it("defines --font-display custom property for headings", () => {
    expect(css).toContain("--font-display:");
  });

  it("@theme block maps --font-display", () => {
    const themeBlock = css.match(/@theme inline\s*\{([\s\S]*?)\n\}/)?.[1] ?? "";
    expect(themeBlock).toContain("--font-display:");
  });

  it("defines heading typography styles (h1-h4) with display font", () => {
    expect(css).toMatch(/h1\s*\{[\s\S]*?font-family:\s*var\(--font-display\)/);
    expect(css).toMatch(/h2\s*\{[\s\S]*?font-family:\s*var\(--font-display\)/);
    expect(css).toMatch(/h3\s*\{[\s\S]*?font-family:\s*var\(--font-display\)/);
    expect(css).toMatch(/h4\s*\{[\s\S]*?font-family:\s*var\(--font-display\)/);
  });

  it("h1 uses fluid font size with clamp (max 2.25rem)", () => {
    expect(css).toMatch(/h1\s*\{[\s\S]*?font-size:\s*clamp\(.*2\.25rem\)/);
  });

  it("headings use line-height 1.2", () => {
    expect(css).toMatch(/h1\s*\{[\s\S]*?line-height:\s*1\.2/);
  });
});

// ─── FR-5: Gradient Accents ─────────────────────────────────────────────────

describe("FR-5: Gradient Accents", () => {
  it("defines gradient custom properties", () => {
    expect(css).toContain("--gradient-primary:");
    expect(css).toContain("--gradient-accent:");
    expect(css).toContain("--gradient-surface:");
  });

  it("defines .bg-gradient-primary utility", () => {
    expect(css).toContain(".bg-gradient-primary");
  });

  it("defines .bg-gradient-accent utility", () => {
    expect(css).toContain(".bg-gradient-accent");
  });

  it("defines .text-gradient utility with background-clip", () => {
    expect(css).toContain(".text-gradient");
    expect(css).toContain("background-clip");
  });
});

// ─── FR-6: Elevation System ─────────────────────────────────────────────────

describe("FR-6: Elevation System", () => {
  it("defines shadow scale tokens (xs through 2xl)", () => {
    expect(css).toContain("--shadow-xs:");
    expect(css).toContain("--shadow-sm:");
    expect(css).toContain("--shadow-md:");
    expect(css).toContain("--shadow-lg:");
    expect(css).toContain("--shadow-xl:");
    expect(css).toContain("--shadow-2xl:");
  });

  it("maps shadow tokens to Tailwind via @theme", () => {
    const themeBlock = css.match(/@theme inline\s*\{([\s\S]*?)\n\}/)?.[1] ?? "";
    expect(themeBlock).toContain("--shadow-");
  });

  it("dark mode defines its own shadow values", () => {
    const darkBlock = css.match(/\.dark\s*\{([\s\S]*?)\n\}/)?.[1] ?? "";
    expect(darkBlock).toContain("--shadow-xs:");
    expect(darkBlock).toContain("--shadow-2xl:");
  });
});

// ─── FR-7: Focus Rings & Transitions ────────────────────────────────────────

describe("FR-7: Focus Rings & Transitions", () => {
  it("ring color uses primary-derived indigo/violet (not gray)", () => {
    const rootBlock = css.match(/:root\s*\{([\s\S]*?)\n\}/)?.[1] ?? "";
    const ringMatch = rootBlock.match(/--ring:\s*oklch\(([^)]+)\)/);
    expect(ringMatch).toBeTruthy();
    const parts = ringMatch![1].split(/\s+/);
    expect(parseFloat(parts[1])).toBeGreaterThan(0);
  });

  it("applies 200ms transitions to interactive elements", () => {
    expect(css).toContain("transition");
    expect(css).toContain("200ms");
    expect(css).toMatch(/button/);
    expect(css).toMatch(/input/);
  });

  it("respects prefers-reduced-motion", () => {
    expect(css).toContain("prefers-reduced-motion");
    expect(css).toContain("reduce");
  });
});

// ─── Regression: Existing Components ────────────────────────────────────────

describe("Regression: Existing shadcn/ui components", () => {
  it("Button renders without errors (default variant)", () => {
    render(<Button data-testid="btn">Click me</Button>);
    expect(screen.getByTestId("btn")).toBeInTheDocument();
    expect(screen.getByTestId("btn")).toHaveTextContent("Click me");
  });

  it("Button renders all variants without errors", () => {
    const variants = [
      "default",
      "outline",
      "secondary",
      "ghost",
      "destructive",
      "link",
    ] as const;
    for (const variant of variants) {
      const { unmount } = render(
        <Button variant={variant} data-testid={`btn-${variant}`}>
          {variant}
        </Button>
      );
      expect(screen.getByTestId(`btn-${variant}`)).toBeInTheDocument();
      unmount();
    }
  });

  it("Input renders without errors", () => {
    render(<Input data-testid="input" placeholder="Type..." />);
    expect(screen.getByTestId("input")).toBeInTheDocument();
  });

  it("GlassCard renders children", () => {
    render(<GlassCard data-testid="gc">Hello</GlassCard>);
    expect(screen.getByTestId("gc")).toHaveTextContent("Hello");
  });
});
