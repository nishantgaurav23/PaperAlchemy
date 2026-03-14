import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { NotificationBell } from "./notification-bell";

vi.mock("next/navigation", () => ({
  usePathname: vi.fn(() => "/"),
}));

describe("NotificationBell", () => {
  it("renders bell button", () => {
    render(<NotificationBell count={3} />);
    expect(screen.getByRole("button", { name: /notification/i })).toBeInTheDocument();
  });

  it("shows badge with count when count > 0", () => {
    render(<NotificationBell count={5} />);
    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("hides badge when count is 0", () => {
    render(<NotificationBell count={0} />);
    expect(screen.queryByTestId("notification-badge")).not.toBeInTheDocument();
  });

  it("shows 99+ when count exceeds 99", () => {
    render(<NotificationBell count={150} />);
    expect(screen.getByText("99+")).toBeInTheDocument();
  });

  it("opens dropdown on click", async () => {
    const user = userEvent.setup();
    render(<NotificationBell count={2} />);

    const bell = screen.getByRole("button", { name: /notification/i });
    await user.click(bell);

    expect(screen.getByText(/notifications/i)).toBeInTheDocument();
  });

  it("shows placeholder notification items in dropdown", async () => {
    const user = userEvent.setup();
    render(<NotificationBell count={2} />);

    const bell = screen.getByRole("button", { name: /notification/i });
    await user.click(bell);

    // Should show some notification placeholder items
    expect(screen.getByText(/new paper/i)).toBeInTheDocument();
  });
});
