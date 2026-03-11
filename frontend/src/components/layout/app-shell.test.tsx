import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { AppShell } from "./app-shell";

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(() => "/search"),
}));

vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "light", setTheme: vi.fn() }),
}));

Object.defineProperty(window, "localStorage", {
  value: {
    getItem: vi.fn(() => null),
    setItem: vi.fn(),
  },
});

describe("AppShell", () => {
  it("renders children content", () => {
    render(
      <AppShell>
        <div data-testid="page-content">Hello</div>
      </AppShell>
    );
    expect(screen.getByTestId("page-content")).toBeInTheDocument();
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });

  it("renders sidebar", () => {
    render(
      <AppShell>
        <div>Content</div>
      </AppShell>
    );
    expect(screen.getByRole("navigation", { name: /main/i })).toBeInTheDocument();
  });

  it("renders header", () => {
    render(
      <AppShell>
        <div>Content</div>
      </AppShell>
    );
    expect(screen.getByRole("banner")).toBeInTheDocument();
  });

  it("renders breadcrumbs in header", () => {
    render(
      <AppShell>
        <div>Content</div>
      </AppShell>
    );
    expect(screen.getByRole("navigation", { name: /breadcrumb/i })).toBeInTheDocument();
  });

  it("renders theme toggle in header", () => {
    render(
      <AppShell>
        <div>Content</div>
      </AppShell>
    );
    expect(screen.getByRole("button", { name: /toggle theme/i })).toBeInTheDocument();
  });

  it("renders main content area", () => {
    render(
      <AppShell>
        <div>Content</div>
      </AppShell>
    );
    expect(screen.getByRole("main")).toBeInTheDocument();
  });
});
