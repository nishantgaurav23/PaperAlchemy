import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { Breadcrumbs } from "./breadcrumbs";
import { usePathname } from "next/navigation";

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(() => "/"),
}));

const mockUsePathname = vi.mocked(usePathname);

describe("Breadcrumbs", () => {
  it("renders nothing for root path", () => {
    mockUsePathname.mockReturnValue("/");

    const { container } = render(<Breadcrumbs />);
    expect(container.querySelector("nav")).toBeNull();
  });

  it("renders single breadcrumb for top-level route", () => {
    mockUsePathname.mockReturnValue("/search");

    render(<Breadcrumbs />);
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByText("Search")).toBeInTheDocument();
  });

  it("renders nested breadcrumbs", () => {
    mockUsePathname.mockReturnValue("/papers/abc-123");

    render(<Breadcrumbs />);
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByText("Papers")).toBeInTheDocument();
    expect(screen.getByText("Abc-123")).toBeInTheDocument();
  });

  it("renders Home as a link to /", () => {
    mockUsePathname.mockReturnValue("/search");

    render(<Breadcrumbs />);
    const homeLink = screen.getByRole("link", { name: /home/i });
    expect(homeLink).toHaveAttribute("href", "/");
  });

  it("renders intermediate segments as links", () => {
    mockUsePathname.mockReturnValue("/papers/abc-123");

    render(<Breadcrumbs />);
    const papersLink = screen.getByRole("link", { name: /papers/i });
    expect(papersLink).toHaveAttribute("href", "/papers");
  });

  it("renders last segment as text (not a link)", () => {
    mockUsePathname.mockReturnValue("/search");

    render(<Breadcrumbs />);
    const searchText = screen.getByText("Search");
    expect(searchText.closest("a")).toBeNull();
  });

  it("has aria-label for accessibility", () => {
    mockUsePathname.mockReturnValue("/search");

    render(<Breadcrumbs />);
    expect(screen.getByRole("navigation", { name: /breadcrumb/i })).toBeInTheDocument();
  });

  it("capitalizes segment names", () => {
    mockUsePathname.mockReturnValue("/collections");

    render(<Breadcrumbs />);
    expect(screen.getByText("Collections")).toBeInTheDocument();
  });
});
