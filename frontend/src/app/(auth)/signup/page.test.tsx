import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import SignupPage from "./page";
import { useAuthStore } from "@/lib/auth/store";

const mockPush = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
}));

vi.mock("sonner", () => ({
  toast: {
    error: vi.fn(),
    success: vi.fn(),
  },
}));

describe("SignupPage", () => {
  beforeEach(() => {
    mockPush.mockReset();
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
  });

  it("renders all form fields", () => {
    render(<SignupPage />);

    expect(screen.getByLabelText(/name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^password$/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/confirm password/i)).toBeInTheDocument();
  });

  it("renders submit button", () => {
    render(<SignupPage />);

    expect(screen.getByRole("button", { name: /create account/i })).toBeInTheDocument();
  });

  it("renders link to login", () => {
    render(<SignupPage />);

    expect(screen.getByRole("link", { name: /sign in/i })).toBeInTheDocument();
  });

  it("shows error for password without number", async () => {
    const user = userEvent.setup();
    render(<SignupPage />);

    await user.type(screen.getByLabelText(/name/i), "Test");
    await user.type(screen.getByLabelText(/email/i), "test@example.com");
    await user.type(screen.getByLabelText(/^password$/i), "password");
    await user.type(screen.getByLabelText(/confirm password/i), "password");
    await user.click(screen.getByRole("button", { name: /create account/i }));

    await waitFor(() => {
      expect(screen.getByText(/must include.*number/i)).toBeInTheDocument();
    });
  });

  it("shows error for password mismatch", async () => {
    const user = userEvent.setup();
    render(<SignupPage />);

    await user.type(screen.getByLabelText(/name/i), "Test");
    await user.type(screen.getByLabelText(/email/i), "test@example.com");
    await user.type(screen.getByLabelText(/^password$/i), "password1");
    await user.type(screen.getByLabelText(/confirm password/i), "password2");
    await user.click(screen.getByRole("button", { name: /create account/i }));

    await waitFor(() => {
      expect(screen.getByText(/passwords don.t match/i)).toBeInTheDocument();
    });
  });

  it("calls signup on valid submit", async () => {
    const signupSpy = vi.fn().mockResolvedValue(undefined);
    useAuthStore.setState({ signup: signupSpy } as unknown as ReturnType<typeof useAuthStore.getState>);

    const user = userEvent.setup();
    render(<SignupPage />);

    await user.type(screen.getByLabelText(/name/i), "Test User");
    await user.type(screen.getByLabelText(/email/i), "test@example.com");
    await user.type(screen.getByLabelText(/^password$/i), "password1");
    await user.type(screen.getByLabelText(/confirm password/i), "password1");
    await user.click(screen.getByRole("button", { name: /create account/i }));

    await waitFor(() => {
      expect(signupSpy).toHaveBeenCalledWith({
        email: "test@example.com",
        password: "password1",
        name: "Test User",
      });
    });
  });

  it("redirects to / after successful signup", async () => {
    const signupSpy = vi.fn().mockResolvedValue(undefined);
    useAuthStore.setState({ signup: signupSpy } as unknown as ReturnType<typeof useAuthStore.getState>);

    const user = userEvent.setup();
    render(<SignupPage />);

    await user.type(screen.getByLabelText(/name/i), "Test User");
    await user.type(screen.getByLabelText(/email/i), "test@example.com");
    await user.type(screen.getByLabelText(/^password$/i), "password1");
    await user.type(screen.getByLabelText(/confirm password/i), "password1");
    await user.click(screen.getByRole("button", { name: /create account/i }));

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith("/");
    });
  });
});
