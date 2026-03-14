import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { Header } from "./header";

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(() => "/search"),
  useRouter: vi.fn(() => ({ push: vi.fn() })),
}));

vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "light", setTheme: vi.fn() }),
}));

describe("Header", () => {
  it("renders the header element", () => {
    render(<Header />);
    expect(screen.getByRole("banner")).toBeInTheDocument();
  });

  it("renders breadcrumbs", () => {
    render(<Header />);
    expect(screen.getByRole("navigation", { name: /breadcrumb/i })).toBeInTheDocument();
  });

  it("renders theme toggle", () => {
    render(<Header />);
    expect(screen.getByRole("button", { name: /toggle theme/i })).toBeInTheDocument();
  });

  it("renders notification bell", () => {
    render(<Header />);
    expect(screen.getByRole("button", { name: /notification/i })).toBeInTheDocument();
  });

  it("renders mobile menu button on small screens", () => {
    render(<Header onMobileMenuToggle={() => {}} showMobileMenu />);
    expect(screen.getByRole("button", { name: /open menu/i })).toBeInTheDocument();
  });

  it("does not render mobile menu button by default", () => {
    render(<Header />);
    expect(screen.queryByRole("button", { name: /open menu/i })).not.toBeInTheDocument();
  });
});
