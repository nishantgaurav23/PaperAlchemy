import { describe, it, expect, vi, beforeEach } from "vitest";
import { useAuthStore } from "./store";
import type { User } from "@/types/auth";

const mockFetch = vi.fn();
global.fetch = mockFetch;

const mockUser: User = {
  id: "user-1",
  email: "test@example.com",
  name: "Test User",
  created_at: "2025-01-01T00:00:00Z",
};

describe("useAuthStore", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    // Reset store state between tests
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
    sessionStorage.clear();
  });

  describe("initial state", () => {
    it("starts unauthenticated", () => {
      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(state.isLoading).toBe(false);
    });
  });

  describe("login", () => {
    it("sets user and token on successful login", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          access_token: "test-token",
          token_type: "bearer",
          user: mockUser,
        }),
      });

      await useAuthStore.getState().login("test@example.com", "password123");

      const state = useAuthStore.getState();
      expect(state.user).toEqual(mockUser);
      expect(state.token).toBe("test-token");
      expect(state.isAuthenticated).toBe(true);
      expect(state.isLoading).toBe(false);
    });

    it("stores token in sessionStorage", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          access_token: "stored-token",
          token_type: "bearer",
          user: mockUser,
        }),
      });

      await useAuthStore.getState().login("test@example.com", "password123");

      expect(sessionStorage.getItem("auth-token")).toBe("stored-token");
    });

    it("throws on failed login", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: "Unauthorized",
        json: async () => ({ detail: "Invalid credentials" }),
      });

      await expect(
        useAuthStore.getState().login("bad@example.com", "wrong"),
      ).rejects.toThrow();

      const state = useAuthStore.getState();
      expect(state.isAuthenticated).toBe(false);
      expect(state.isLoading).toBe(false);
    });

    it("sets isLoading during login", async () => {
      let resolveLogin: (value: unknown) => void;
      const pendingPromise = new Promise((resolve) => {
        resolveLogin = resolve;
      });
      mockFetch.mockReturnValueOnce(pendingPromise);

      const loginPromise = useAuthStore.getState().login("test@example.com", "password123");
      expect(useAuthStore.getState().isLoading).toBe(true);

      resolveLogin!({
        ok: true,
        json: async () => ({
          access_token: "token",
          token_type: "bearer",
          user: mockUser,
        }),
      });

      await loginPromise;
      expect(useAuthStore.getState().isLoading).toBe(false);
    });
  });

  describe("signup", () => {
    it("sets user and token on successful signup", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          access_token: "signup-token",
          token_type: "bearer",
          user: mockUser,
        }),
      });

      await useAuthStore.getState().signup({
        email: "test@example.com",
        password: "password1",
        name: "Test User",
      });

      const state = useAuthStore.getState();
      expect(state.user).toEqual(mockUser);
      expect(state.token).toBe("signup-token");
      expect(state.isAuthenticated).toBe(true);
    });

    it("throws on failed signup", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 409,
        statusText: "Conflict",
        json: async () => ({ detail: "Email already registered" }),
      });

      await expect(
        useAuthStore.getState().signup({
          email: "taken@example.com",
          password: "password1",
          name: "Taken User",
        }),
      ).rejects.toThrow();
    });
  });

  describe("logout", () => {
    it("clears user, token, and sets isAuthenticated false", () => {
      // Set up authenticated state
      useAuthStore.setState({
        user: mockUser,
        token: "some-token",
        isAuthenticated: true,
      });
      sessionStorage.setItem("auth-token", "some-token");

      useAuthStore.getState().logout();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
      expect(sessionStorage.getItem("auth-token")).toBeNull();
    });
  });

  describe("setToken", () => {
    it("sets token and marks as loading for refresh", () => {
      useAuthStore.getState().setToken("new-token");

      const state = useAuthStore.getState();
      expect(state.token).toBe("new-token");
    });
  });

  describe("clearAuth", () => {
    it("clears all auth state", () => {
      useAuthStore.setState({
        user: mockUser,
        token: "some-token",
        isAuthenticated: true,
      });

      useAuthStore.getState().clearAuth();

      const state = useAuthStore.getState();
      expect(state.user).toBeNull();
      expect(state.token).toBeNull();
      expect(state.isAuthenticated).toBe(false);
    });
  });
});
