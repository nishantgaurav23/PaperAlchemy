import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { ProgressIndicator } from "./progress-indicator";

describe("ProgressIndicator", () => {
  it("renders with determinate progress", () => {
    render(<ProgressIndicator value={50} />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toBeInTheDocument();
    expect(bar.getAttribute("aria-valuenow")).toBe("50");
  });

  it("renders fill bar at correct width percentage", () => {
    render(<ProgressIndicator value={75} />);
    const fill = screen.getByTestId("progress-fill");
    expect(fill.style.width).toBe("75%");
  });

  it("renders indeterminate state when no value provided", () => {
    render(<ProgressIndicator />);
    const fill = screen.getByTestId("progress-fill");
    expect(fill.className).toContain("animate-");
  });

  it("handles 0% value", () => {
    render(<ProgressIndicator value={0} />);
    const fill = screen.getByTestId("progress-fill");
    expect(fill.style.width).toBe("0%");
  });

  it("handles 100% value", () => {
    render(<ProgressIndicator value={100} />);
    const fill = screen.getByTestId("progress-fill");
    expect(fill.style.width).toBe("100%");
  });

  it("shows error state with error variant", () => {
    render(<ProgressIndicator value={30} variant="error" />);
    const fill = screen.getByTestId("progress-fill");
    expect(fill.className).toContain("bg-destructive");
  });

  it("shows success state with success variant", () => {
    render(<ProgressIndicator value={100} variant="success" />);
    const fill = screen.getByTestId("progress-fill");
    expect(fill.className).toContain("bg-success");
  });

  it("renders label when provided", () => {
    render(<ProgressIndicator value={50} label="Uploading..." />);
    expect(screen.getByText("Uploading...")).toBeInTheDocument();
  });

  it("renders percentage text when showPercentage is true", () => {
    render(<ProgressIndicator value={65} showPercentage />);
    expect(screen.getByText("65%")).toBeInTheDocument();
  });

  it("applies custom className", () => {
    render(<ProgressIndicator value={50} className="w-full" />);
    const bar = screen.getByRole("progressbar");
    expect(bar.className).toContain("w-full");
  });
});
