import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import {
  SkeletonCard,
  SkeletonText,
  SkeletonChart,
  SkeletonList,
} from "./skeleton-shimmer";

describe("SkeletonCard", () => {
  it("renders with animate-pulse class", () => {
    render(<SkeletonCard data-testid="skel-card" />);
    const el = screen.getByTestId("skel-card");
    expect(el.className).toContain("animate-pulse");
  });

  it("renders card shape with rounded corners", () => {
    render(<SkeletonCard data-testid="skel-card" />);
    const el = screen.getByTestId("skel-card");
    expect(el.className).toContain("rounded");
  });

  it("accepts custom className", () => {
    render(<SkeletonCard data-testid="skel-card" className="h-40" />);
    const el = screen.getByTestId("skel-card");
    expect(el.className).toContain("h-40");
  });
});

describe("SkeletonText", () => {
  it("renders multiple lines", () => {
    render(<SkeletonText lines={3} data-testid="skel-text" />);
    const container = screen.getByTestId("skel-text");
    const lines = container.querySelectorAll("[data-slot='skeleton-line']");
    expect(lines.length).toBe(3);
  });

  it("renders single line by default", () => {
    render(<SkeletonText data-testid="skel-text" />);
    const container = screen.getByTestId("skel-text");
    const lines = container.querySelectorAll("[data-slot='skeleton-line']");
    expect(lines.length).toBe(1);
  });

  it("last line is shorter than others", () => {
    render(<SkeletonText lines={3} data-testid="skel-text" />);
    const container = screen.getByTestId("skel-text");
    const lines = container.querySelectorAll("[data-slot='skeleton-line']");
    const lastLine = lines[lines.length - 1];
    expect(lastLine.className).toContain("w-3/4");
  });
});

describe("SkeletonChart", () => {
  it("renders with shimmer class", () => {
    render(<SkeletonChart data-testid="skel-chart" />);
    const el = screen.getByTestId("skel-chart");
    expect(el.className).toContain("shimmer");
  });

  it("has chart-appropriate aspect ratio", () => {
    render(<SkeletonChart data-testid="skel-chart" />);
    const el = screen.getByTestId("skel-chart");
    expect(el.className).toContain("aspect-video");
  });
});

describe("SkeletonList", () => {
  it("renders specified number of items", () => {
    render(<SkeletonList count={5} data-testid="skel-list" />);
    const container = screen.getByTestId("skel-list");
    const items = container.querySelectorAll("[data-slot='skeleton-item']");
    expect(items.length).toBe(5);
  });

  it("defaults to 3 items", () => {
    render(<SkeletonList data-testid="skel-list" />);
    const container = screen.getByTestId("skel-list");
    const items = container.querySelectorAll("[data-slot='skeleton-item']");
    expect(items.length).toBe(3);
  });
});
