import { describe, it, expect, vi, beforeEach } from "vitest";
import { createAuthenticatedFetch } from "./interceptor";
import { useAuthStore } from "./store";

const mockFetch = vi.fn();

describe("createAuthenticatedFetch", () => {
  let authFetch: typeof fetch;

  beforeEach(() => {
    mockFetch.mockReset();
    useAuthStore.setState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
    });
    authFetch = createAuthenticatedFetch(mockFetch);
  });

  it("attaches Bearer token when authenticated", async () => {
    useAuthStore.setState({ token: "my-token", isAuthenticated: true });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({}),
    });

    await authFetch("http://localhost:8002/api/v1/papers");

    const calledHeaders = mockFetch.mock.calls[0][1]?.headers;
    expect(calledHeaders).toBeDefined();
    expect(new Headers(calledHeaders).get("Authorization")).toBe("Bearer my-token");
  });

  it("does not attach header when no token", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({}),
    });

    await authFetch("http://localhost:8002/api/v1/papers");

    const calledHeaders = mockFetch.mock.calls[0][1]?.headers;
    if (calledHeaders) {
      expect(new Headers(calledHeaders).get("Authorization")).toBeNull();
    }
  });

  it("clears auth state on 401 response", async () => {
    useAuthStore.setState({
      user: { id: "1", email: "a@b.com", name: "A", created_at: "2025-01-01T00:00:00Z" },
      token: "expired-token",
      isAuthenticated: true,
    });

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      statusText: "Unauthorized",
    });

    const response = await authFetch("http://localhost:8002/api/v1/papers");
    expect(response.status).toBe(401);

    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(false);
    expect(state.token).toBeNull();
    expect(state.user).toBeNull();
  });

  it("does NOT clear auth state on 403 response", async () => {
    useAuthStore.setState({
      user: { id: "1", email: "a@b.com", name: "A", created_at: "2025-01-01T00:00:00Z" },
      token: "valid-token",
      isAuthenticated: true,
    });

    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 403,
      statusText: "Forbidden",
    });

    const response = await authFetch("http://localhost:8002/api/v1/admin");
    expect(response.status).toBe(403);

    const state = useAuthStore.getState();
    expect(state.isAuthenticated).toBe(true);
    expect(state.token).toBe("valid-token");
  });

  it("preserves existing headers from request", async () => {
    useAuthStore.setState({ token: "my-token", isAuthenticated: true });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({}),
    });

    await authFetch("http://localhost:8002/api/v1/papers", {
      headers: { "Content-Type": "application/json" },
    });

    const calledHeaders = new Headers(mockFetch.mock.calls[0][1]?.headers);
    expect(calledHeaders.get("Authorization")).toBe("Bearer my-token");
    expect(calledHeaders.get("Content-Type")).toBe("application/json");
  });
});
