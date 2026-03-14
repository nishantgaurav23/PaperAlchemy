import { describe, it, expect, vi, beforeEach } from "vitest";
import { MOCK_RESPONSE } from "./upload";

describe("uploadPdf", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("MOCK_RESPONSE has correct paper fields", () => {
    expect(MOCK_RESPONSE.paper.title).toBe("Attention Is All You Need");
    expect(MOCK_RESPONSE.summary.objective).toBeTruthy();
    expect(MOCK_RESPONSE.highlights.novel_contributions.length).toBeGreaterThan(0);
    expect(MOCK_RESPONSE.methodology.datasets.length).toBeGreaterThan(0);
  });

  it("mock response has required fields", () => {
    expect(MOCK_RESPONSE.paper.id).toBeTruthy();
    expect(MOCK_RESPONSE.paper.title).toBeTruthy();
    expect(MOCK_RESPONSE.paper.authors.length).toBeGreaterThan(0);
    expect(MOCK_RESPONSE.summary.objective).toBeTruthy();
    expect(MOCK_RESPONSE.summary.method).toBeTruthy();
    expect(MOCK_RESPONSE.summary.key_findings).toBeTruthy();
    expect(MOCK_RESPONSE.summary.contribution).toBeTruthy();
    expect(MOCK_RESPONSE.summary.limitations).toBeTruthy();
    expect(MOCK_RESPONSE.highlights.novel_contributions.length).toBeGreaterThan(0);
    expect(MOCK_RESPONSE.highlights.important_findings.length).toBeGreaterThan(0);
    expect(MOCK_RESPONSE.highlights.practical_implications.length).toBeGreaterThan(0);
    expect(MOCK_RESPONSE.methodology.approach).toBeTruthy();
    expect(MOCK_RESPONSE.methodology.datasets.length).toBeGreaterThan(0);
    expect(MOCK_RESPONSE.methodology.baselines.length).toBeGreaterThan(0);
    expect(MOCK_RESPONSE.methodology.results).toBeTruthy();
  });
});
