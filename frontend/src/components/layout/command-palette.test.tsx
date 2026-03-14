import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { CommandPalette } from "./command-palette";

const mockPush = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({ push: mockPush })),
  usePathname: vi.fn(() => "/"),
}));

describe("CommandPalette", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders nothing when closed", () => {
    render(<CommandPalette open={false} onOpenChange={() => {}} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders dialog when open", () => {
    render(<CommandPalette open={true} onOpenChange={() => {}} />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("renders search input", () => {
    render(<CommandPalette open={true} onOpenChange={() => {}} />);
    const input = screen.getByPlaceholderText(/search|type a command/i);
    expect(input).toBeInTheDocument();
  });

  it("renders grouped results: Pages", () => {
    render(<CommandPalette open={true} onOpenChange={() => {}} />);
    expect(screen.getByText("Pages")).toBeInTheDocument();
    expect(screen.getByText("Search")).toBeInTheDocument();
    expect(screen.getByText("Chat")).toBeInTheDocument();
  });

  it("renders grouped results: Actions", () => {
    render(<CommandPalette open={true} onOpenChange={() => {}} />);
    expect(screen.getByText("Actions")).toBeInTheDocument();
  });

  it("filters items based on search query", async () => {
    const user = userEvent.setup();
    render(<CommandPalette open={true} onOpenChange={() => {}} />);

    const input = screen.getByPlaceholderText(/search|type a command/i);
    await user.type(input, "chat");

    // Chat should be visible, other non-matching items hidden
    expect(screen.getByText("Chat")).toBeInTheDocument();
  });

  it("shows empty state when no results match", async () => {
    const user = userEvent.setup();
    render(<CommandPalette open={true} onOpenChange={() => {}} />);

    const input = screen.getByPlaceholderText(/search|type a command/i);
    await user.type(input, "xyznonexistent");

    expect(screen.getByText(/no results/i)).toBeInTheDocument();
  });

  it("navigates when a page item is selected", async () => {
    const onOpenChange = vi.fn();
    const user = userEvent.setup();
    render(<CommandPalette open={true} onOpenChange={onOpenChange} />);

    const searchItem = screen.getByText("Search");
    await user.click(searchItem);

    expect(mockPush).toHaveBeenCalledWith("/search");
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("calls onOpenChange(false) when closing", () => {
    const onOpenChange = vi.fn();
    render(<CommandPalette open={true} onOpenChange={onOpenChange} />);

    // Press Escape
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});
