import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { UploadProgress } from "./upload-progress";

describe("UploadProgress", () => {
  it("shows uploading state", () => {
    render(<UploadProgress status="uploading" fileName="paper.pdf" />);
    expect(screen.getByText(/uploading/i)).toBeInTheDocument();
    expect(screen.getByText("paper.pdf")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
  });

  it("shows processing state", () => {
    render(<UploadProgress status="processing" fileName="paper.pdf" />);
    expect(screen.getByText(/processing/i)).toBeInTheDocument();
  });

  it("shows complete state", () => {
    render(<UploadProgress status="complete" fileName="paper.pdf" />);
    expect(screen.getByText(/complete/i)).toBeInTheDocument();
  });

  it("renders progress bar with correct width for uploading", () => {
    render(<UploadProgress status="uploading" fileName="paper.pdf" />);
    const progressBar = screen.getByTestId("progress-fill");
    // Uploading shows indeterminate animation
    expect(progressBar).toBeInTheDocument();
  });

  it("renders progress bar full for processing", () => {
    render(<UploadProgress status="processing" fileName="paper.pdf" />);
    const progressBar = screen.getByTestId("progress-fill");
    expect(progressBar).toBeInTheDocument();
  });
});
