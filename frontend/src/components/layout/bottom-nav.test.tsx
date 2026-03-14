import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { BottomNav } from "./bottom-nav";

// Mock next/navigation
const mockPathname = vi.fn(() => "/search");
vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname(),
}));

describe("BottomNav", () => {
  beforeEach(() => {
    mockPathname.mockReturnValue("/search");
  });

  it("renders a navigation element", () => {
    render(<BottomNav />);
    expect(screen.getByRole("navigation", { name: /bottom/i })).toBeInTheDocument();
  });

  it("renders all 6 nav items", () => {
    render(<BottomNav />);
    expect(screen.getByText("Search")).toBeInTheDocument();
    expect(screen.getByText("Chat")).toBeInTheDocument();
    expect(screen.getByText("Upload")).toBeInTheDocument();
    expect(screen.getByText("Papers")).toBeInTheDocument();
    expect(screen.getByText("Collections")).toBeInTheDocument();
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
  });

  it("renders nav items as links", () => {
    render(<BottomNav />);
    const links = screen.getAllByRole("link");
    expect(links.length).toBe(6);
    expect(links[0]).toHaveAttribute("href", "/search");
    expect(links[1]).toHaveAttribute("href", "/chat");
  });

  it("highlights the active route", () => {
    mockPathname.mockReturnValue("/chat");
    render(<BottomNav />);
    const chatLink = screen.getByRole("link", { name: /chat/i });
    expect(chatLink).toHaveAttribute("data-active", "true");
  });

  it("does not highlight inactive routes", () => {
    mockPathname.mockReturnValue("/chat");
    render(<BottomNav />);
    const searchLink = screen.getByRole("link", { name: /search/i });
    expect(searchLink).toHaveAttribute("data-active", "false");
  });

  it("has fixed positioning at the bottom", () => {
    render(<BottomNav />);
    const nav = screen.getByRole("navigation", { name: /bottom/i });
    expect(nav.className).toContain("fixed");
    expect(nav.className).toContain("bottom-0");
  });

  it("is hidden on md+ screens via CSS class", () => {
    render(<BottomNav />);
    const nav = screen.getByRole("navigation", { name: /bottom/i });
    expect(nav.className).toContain("md:hidden");
  });

  it("each nav item has min 44px touch target", () => {
    render(<BottomNav />);
    const links = screen.getAllByRole("link");
    for (const link of links) {
      expect(link.className).toContain("min-h-[44px]");
    }
  });
});
