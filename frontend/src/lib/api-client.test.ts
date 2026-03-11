import { describe, it, expect, vi, beforeEach } from "vitest";
import { apiClient, ApiError } from "./api-client";

const mockFetch = vi.fn();
global.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
});

describe("apiClient", () => {
  describe("get", () => {
    it("makes GET request with correct URL", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: "ok" }),
      });

      const result = await apiClient.get("/api/v1/health");

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/v1/health",
        expect.objectContaining({ method: "GET" }),
      );
      expect(result).toEqual({ status: "ok" });
    });
  });

  describe("post", () => {
    it("makes POST request with JSON body", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: "123" }),
      });

      const body = { query: "transformers" };
      const result = await apiClient.post("/api/v1/search", body);

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/v1/search",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify(body),
        }),
      );
      expect(result).toEqual({ id: "123" });
    });

    it("makes POST request without body", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      await apiClient.post("/api/v1/action");

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/v1/action",
        expect.objectContaining({
          method: "POST",
          body: undefined,
        }),
      );
    });
  });

  describe("error handling", () => {
    it("throws ApiError on non-ok response with JSON body", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: "Not Found",
        json: async () => ({ detail: "Resource not found" }),
      });

      await expect(apiClient.get("/api/v1/missing")).rejects.toThrow(ApiError);
      await expect(
        apiClient.get("/api/v1/missing").catch((e) => {
          expect(e).toBeInstanceOf(ApiError);
          expect(e.status).toBe(404);
          expect(e.body).toEqual({ detail: "Resource not found" });
          throw e;
        }),
      ).rejects.toThrow();
    });

    it("throws ApiError on non-ok response with text body", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: async () => {
          throw new Error("not json");
        },
        text: async () => "Server error",
      });

      await expect(apiClient.get("/api/v1/broken")).rejects.toThrow(ApiError);
    });

    it("handles fetch abort (timeout)", async () => {
      mockFetch.mockImplementation(
        () => new Promise((_resolve, reject) => {
          setTimeout(() => reject(new DOMException("Aborted", "AbortError")), 50);
        }),
      );

      await expect(
        apiClient.get("/api/v1/slow", { timeout: 10 }),
      ).rejects.toThrow();
    });
  });

  describe("put", () => {
    it("makes PUT request", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ updated: true }),
      });

      await apiClient.put("/api/v1/item/1", { name: "updated" });

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/v1/item/1",
        expect.objectContaining({ method: "PUT" }),
      );
    });
  });

  describe("delete", () => {
    it("makes DELETE request", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ deleted: true }),
      });

      await apiClient.delete("/api/v1/item/1");

      expect(mockFetch).toHaveBeenCalledWith(
        "http://localhost:8000/api/v1/item/1",
        expect.objectContaining({ method: "DELETE" }),
      );
    });
  });
});
