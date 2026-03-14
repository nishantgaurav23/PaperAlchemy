import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProtectedRoute } from "./protected-route";
import { useAuthStore } from "@/lib/auth/store";

const mockPush = vi.fn();
const mockPathname = "/dashboard";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
  usePathname: () => mockPathname,
}));

describe("ProtectedRoute", () => {
  beforeEach(() => {
    mockPush.mockReset();
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
  });

  it("renders children when authenticated", () => {
    useAuthStore.setState({
      user: { id: "1", email: "a@b.com", name: "A", created_at: "2025-01-01T00:00:00Z" },
      token: "token",
      isAuthenticated: true,
      isLoading: false,
    });

    render(
      <ProtectedRoute>
        <div>Protected Content</div>
      </ProtectedRoute>,
    );

    expect(screen.getByText("Protected Content")).toBeInTheDocument();
  });

  it("redirects to /login when not authenticated", async () => {
    render(
      <ProtectedRoute>
        <div>Protected Content</div>
      </ProtectedRoute>,
    );

    expect(mockPush).toHaveBeenCalledWith(`/login?redirect=${encodeURIComponent(mockPathname)}`);
    expect(screen.queryByText("Protected Content")).not.toBeInTheDocument();
  });

  it("shows loading state when isLoading", () => {
    useAuthStore.setState({ isLoading: true });

    render(
      <ProtectedRoute>
        <div>Protected Content</div>
      </ProtectedRoute>,
    );

    expect(screen.queryByText("Protected Content")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("renders custom fallback when loading", () => {
    useAuthStore.setState({ isLoading: true });

    render(
      <ProtectedRoute fallback={<div>Custom Loading...</div>}>
        <div>Protected Content</div>
      </ProtectedRoute>,
    );

    expect(screen.getByText("Custom Loading...")).toBeInTheDocument();
  });
});
