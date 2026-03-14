import { create } from "zustand";
import type { AuthState, SignupRequest, User } from "@/types/auth";
import { LoginResponseSchema } from "@/types/auth";

const TOKEN_KEY = "auth-token";
const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8002";

interface AuthActions {
  login: (email: string, password: string) => Promise<void>;
  signup: (data: SignupRequest) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
  setToken: (token: string) => void;
  clearAuth: () => void;
}

function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  return sessionStorage.getItem(TOKEN_KEY);
}

function storeToken(token: string) {
  if (typeof window !== "undefined") {
    sessionStorage.setItem(TOKEN_KEY, token);
  }
}

function removeToken() {
  if (typeof window !== "undefined") {
    sessionStorage.removeItem(TOKEN_KEY);
  }
}

export const useAuthStore = create<AuthState & AuthActions>((set) => ({
  user: null,
  token: getStoredToken(),
  isAuthenticated: false,
  isLoading: false,

  login: async (email: string, password: string) => {
    set({ isLoading: true });
    try {
      const response = await fetch(`${BASE_URL}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Login failed: ${response.statusText}`);
      }

      const data = LoginResponseSchema.parse(await response.json());
      storeToken(data.access_token);
      set({
        user: data.user,
        token: data.access_token,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  signup: async (data: SignupRequest) => {
    set({ isLoading: true });
    try {
      const response = await fetch(`${BASE_URL}/api/v1/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Signup failed: ${response.statusText}`);
      }

      const result = LoginResponseSchema.parse(await response.json());
      storeToken(result.access_token);
      set({
        user: result.user,
        token: result.access_token,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  logout: () => {
    removeToken();
    set({ user: null, token: null, isAuthenticated: false });
  },

  refreshUser: async () => {
    const token = getStoredToken();
    if (!token) return;

    try {
      const response = await fetch(`${BASE_URL}/api/v1/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) {
        removeToken();
        set({ user: null, token: null, isAuthenticated: false });
        return;
      }
      const user: User = await response.json();
      set({ user, token, isAuthenticated: true });
    } catch {
      removeToken();
      set({ user: null, token: null, isAuthenticated: false });
    }
  },

  setToken: (token: string) => {
    storeToken(token);
    set({ token });
  },

  clearAuth: () => {
    removeToken();
    set({ user: null, token: null, isAuthenticated: false, isLoading: false });
  },
}));
