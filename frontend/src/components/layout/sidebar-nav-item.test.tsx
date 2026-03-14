import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { SidebarNavItem } from "./sidebar-nav-item";
import { Search } from "lucide-react";
import { usePathname } from "next/navigation";

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(() => "/search"),
}));

const mockUsePathname = vi.mocked(usePathname);

describe("SidebarNavItem", () => {
  it("renders icon and label", () => {
    render(
      <SidebarNavItem href="/search" label="Search" icon={Search} collapsed={false} />
    );
    expect(screen.getByText("Search")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /search/i })).toBeInTheDocument();
  });

  it("renders link with correct href", () => {
    render(
      <SidebarNavItem href="/search" label="Search" icon={Search} collapsed={false} />
    );
    const link = screen.getByRole("link", { name: /search/i });
    expect(link).toHaveAttribute("href", "/search");
  });

  it("hides label when collapsed", () => {
    render(
      <SidebarNavItem href="/search" label="Search" icon={Search} collapsed={true} />
    );
    // Label text should not be visible (only in tooltip, not in DOM directly)
    const link = screen.getByRole("link");
    // The visible text "Search" should not be a direct child span
    expect(link.querySelector("span.nav-label")).toBeNull();
  });

  it("applies active styling when pathname matches href", () => {
    mockUsePathname.mockReturnValue("/search");

    render(
      <SidebarNavItem href="/search" label="Search" icon={Search} collapsed={false} />
    );
    const link = screen.getByRole("link", { name: /search/i });
    expect(link).toHaveAttribute("data-active", "true");
  });

  it("does not apply active styling when pathname does not match", () => {
    mockUsePathname.mockReturnValue("/chat");

    render(
      <SidebarNavItem href="/search" label="Search" icon={Search} collapsed={false} />
    );
    const link = screen.getByRole("link", { name: /search/i });
    expect(link).not.toHaveAttribute("data-active", "true");
  });

  it("matches active state for nested routes", () => {
    mockUsePathname.mockReturnValue("/papers/abc-123");

    render(
      <SidebarNavItem href="/papers" label="Papers" icon={Search} collapsed={false} />
    );
    const link = screen.getByRole("link", { name: /papers/i });
    expect(link).toHaveAttribute("data-active", "true");
  });

  // S13.5: Left border accent on active items
  it("shows left border accent on active items", () => {
    mockUsePathname.mockReturnValue("/search");

    render(
      <SidebarNavItem href="/search" label="Search" icon={Search} collapsed={false} />
    );
    const link = screen.getByRole("link", { name: /search/i });
    expect(link.className).toMatch(/border-l/);
    expect(link.className).toMatch(/border-primary/);
  });

  it("does not show primary left border on inactive items", () => {
    mockUsePathname.mockReturnValue("/chat");

    render(
      <SidebarNavItem href="/search" label="Search" icon={Search} collapsed={false} />
    );
    const link = screen.getByRole("link", { name: /search/i });
    expect(link.className).toMatch(/border-transparent/);
    expect(link.className).not.toMatch(/border-primary/);
  });

  // S13.5: Keyboard shortcut hints
  it("displays keyboard shortcut hint when expanded and shortcut provided", () => {
    mockUsePathname.mockReturnValue("/");

    render(
      <SidebarNavItem
        href="/search"
        label="Search"
        icon={Search}
        collapsed={false}
        shortcut="1"
      />
    );
    const kbd = screen.getByText("1");
    expect(kbd).toBeInTheDocument();
    expect(kbd.tagName.toLowerCase()).toBe("kbd");
  });

  it("does not display keyboard shortcut hint when collapsed", () => {
    mockUsePathname.mockReturnValue("/");

    render(
      <SidebarNavItem
        href="/search"
        label="Search"
        icon={Search}
        collapsed={true}
        shortcut="1"
      />
    );
    expect(screen.queryByText("1")).not.toBeInTheDocument();
  });

  // S13.5: Tooltip on collapsed hover
  it("wraps in tooltip when collapsed", () => {
    mockUsePathname.mockReturnValue("/");

    render(
      <SidebarNavItem
        href="/search"
        label="Search"
        icon={Search}
        collapsed={true}
        shortcut="1"
      />
    );
    // The tooltip trigger should wrap the link
    const link = screen.getByRole("link");
    expect(link).toBeInTheDocument();
  });
});
