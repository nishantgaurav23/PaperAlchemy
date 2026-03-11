import { describe, it, expect, vi, beforeEach } from "vitest";
import { getDashboardData, MOCK_DASHBOARD_DATA } from "./dashboard";

vi.mock("@/lib/api-client", () => ({
  apiClient: {
    get: vi.fn(),
  },
  ApiError: class ApiError extends Error {
    constructor(
      public status: number,
      public statusText: string,
      public body: unknown,
    ) {
      super(`API Error ${status}: ${statusText}`);
      this.name = "ApiError";
    }
  },
}));

import { apiClient } from "@/lib/api-client";

const mockGet = vi.mocked(apiClient.get);

describe("getDashboardData", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls GET /api/v1/dashboard/stats", async () => {
    mockGet.mockResolvedValue(MOCK_DASHBOARD_DATA);

    await getDashboardData();
    expect(mockGet).toHaveBeenCalledWith("/api/v1/dashboard/stats");
  });

  it("returns API response on success", async () => {
    const apiData = { ...MOCK_DASHBOARD_DATA, stats: { ...MOCK_DASHBOARD_DATA.stats, total_papers: 999 } };
    mockGet.mockResolvedValue(apiData);

    const result = await getDashboardData();
    expect(result.stats.total_papers).toBe(999);
  });

  it("returns mock data when useMock is true", async () => {
    const result = await getDashboardData(true);
    expect(result).toEqual(MOCK_DASHBOARD_DATA);
    expect(mockGet).not.toHaveBeenCalled();
  });

  it("falls back to mock data on API error", async () => {
    mockGet.mockRejectedValue(new Error("Network error"));

    const result = await getDashboardData();
    expect(result).toEqual(MOCK_DASHBOARD_DATA);
  });

  it("mock data has required fields", () => {
    expect(MOCK_DASHBOARD_DATA.stats.total_papers).toBeGreaterThan(0);
    expect(MOCK_DASHBOARD_DATA.categories.length).toBeGreaterThan(0);
    expect(MOCK_DASHBOARD_DATA.timeline.length).toBeGreaterThan(0);
    expect(MOCK_DASHBOARD_DATA.hot_papers.length).toBeGreaterThan(0);
    expect(MOCK_DASHBOARD_DATA.trending_topics.length).toBeGreaterThan(0);
  });
});
