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
    expect(screen.queryByText("Search")).not.toBeInTheDocument();
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
});
