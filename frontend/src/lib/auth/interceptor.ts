import { useAuthStore } from "./store";

export function createAuthenticatedFetch(baseFetch?: typeof fetch): typeof fetch {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const fetchFn = baseFetch ?? globalThis.fetch;
    const token = useAuthStore.getState().token;

    const headers = new Headers(init?.headers);
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }

    const response = await fetchFn(input, { ...init, headers });

    if (response.status === 401) {
      useAuthStore.getState().clearAuth();
    }

    return response;
  };
}
