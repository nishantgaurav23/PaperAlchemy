import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

  // S13.5: Dropdown on Home showing sibling routes
  it("shows dropdown trigger on Home breadcrumb", () => {
    mockUsePathname.mockReturnValue("/search");

    render(<Breadcrumbs />);
    // Home should have a dropdown trigger (chevron/caret)
    const homeLink = screen.getByRole("link", { name: /home/i });
    expect(homeLink).toBeInTheDocument();
  });

  it("shows sibling routes in dropdown when Home is clicked", async () => {
    mockUsePathname.mockReturnValue("/search");
    const user = userEvent.setup();

    render(<Breadcrumbs />);

    // Find the dropdown trigger near Home
    const dropdownTrigger = screen.getByTestId("breadcrumb-dropdown-trigger");
    await user.click(dropdownTrigger);

    // Should show sibling pages
    expect(screen.getByRole("menuitem", { name: /chat/i })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /upload/i })).toBeInTheDocument();
  });
});
