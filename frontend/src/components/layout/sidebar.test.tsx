import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { Sidebar } from "./sidebar";

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(() => "/"),
  useRouter: vi.fn(() => ({ push: vi.fn() })),
}));

vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "light", setTheme: vi.fn() }),
}));

const mockLocalStorage = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(window, "localStorage", { value: mockLocalStorage });

describe("Sidebar", () => {
  beforeEach(() => {
    mockLocalStorage.clear();
    vi.clearAllMocks();
  });

  it("renders all 6 navigation items", () => {
    render(<Sidebar />);
    expect(screen.getByRole("link", { name: /search/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /chat/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /upload/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /papers/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /collections/i })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /dashboard/i })).toBeInTheDocument();
  });

  it("renders PaperAlchemy branding", () => {
    render(<Sidebar />);
    expect(screen.getByText("PaperAlchemy")).toBeInTheDocument();
  });

  it("has a collapse toggle button", () => {
    render(<Sidebar />);
    expect(screen.getByRole("button", { name: /collapse sidebar/i })).toBeInTheDocument();
  });

  it("collapses when toggle is clicked", async () => {
    const user = userEvent.setup();
    render(<Sidebar />);

    const toggleBtn = screen.getByRole("button", { name: /collapse sidebar/i });
    await user.click(toggleBtn);

    expect(screen.getByRole("button", { name: /expand sidebar/i })).toBeInTheDocument();
  });

  it("persists collapsed state to localStorage", async () => {
    const user = userEvent.setup();
    render(<Sidebar />);

    const toggleBtn = screen.getByRole("button", { name: /collapse sidebar/i });
    await user.click(toggleBtn);

    expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
      "sidebar-collapsed",
      "true"
    );
  });

  it("reads initial collapsed state from localStorage", () => {
    mockLocalStorage.getItem.mockReturnValue("true");
    render(<Sidebar />);
    expect(screen.getByRole("button", { name: /expand sidebar/i })).toBeInTheDocument();
  });

  it("renders navigation as a nav element with accessible label", () => {
    render(<Sidebar />);
    expect(screen.getByRole("navigation", { name: /main/i })).toBeInTheDocument();
  });

  // S13.5: Gradient logo mark
  it("renders gradient logo mark container", () => {
    render(<Sidebar />);
    const logoMark = screen.getByTestId("logo-mark");
    expect(logoMark).toBeInTheDocument();
    expect(logoMark.className).toMatch(/gradient/);
  });

  // S13.5: Smooth collapse animation
  it("applies smooth transition classes on sidebar", () => {
    render(<Sidebar />);
    const aside = screen.getByRole("complementary");
    expect(aside.className).toMatch(/transition/);
    expect(aside.className).toMatch(/duration/);
    expect(aside.className).toMatch(/ease/);
  });
});
