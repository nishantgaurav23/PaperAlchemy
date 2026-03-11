import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { MobileNav } from "./mobile-nav";

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(() => "/"),
}));

vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "light", setTheme: vi.fn() }),
}));

describe("MobileNav", () => {
  it("renders nothing when closed", () => {
    const { container } = render(<MobileNav open={false} onClose={() => {}} />);
    expect(container.querySelector("[role='dialog']")).toBeNull();
  });

  it("renders all nav items when open", () => {
    render(<MobileNav open={true} onClose={() => {}} />);
    expect(screen.getByRole("link", { name: /search/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /chat/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /upload/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /papers/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /collections/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /dashboard/i })).toBeInTheDocument();
  });

  it("renders PaperAlchemy branding when open", () => {
    render(<MobileNav open={true} onClose={() => {}} />);
    expect(screen.getByText("PaperAlchemy")).toBeInTheDocument();
  });

  it("calls onClose when close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<MobileNav open={true} onClose={onClose} />);
    const closeBtn = screen.getByRole("button", { name: /close menu/i });
    await user.click(closeBtn);

    expect(onClose).toHaveBeenCalledOnce();
  });

  it("calls onClose when overlay backdrop is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<MobileNav open={true} onClose={onClose} />);
    const backdrop = screen.getByTestId("mobile-nav-backdrop");
    await user.click(backdrop);

    expect(onClose).toHaveBeenCalledOnce();
  });

  it("renders a dialog with accessible label", () => {
    render(<MobileNav open={true} onClose={() => {}} />);
    expect(screen.getByRole("dialog", { name: /navigation menu/i })).toBeInTheDocument();
  });
});
