import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { DropZone } from "./drop-zone";

describe("DropZone", () => {
  const mockOnFileSelect = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders drop zone with prompt text", () => {
    render(<DropZone onFileSelect={mockOnFileSelect} />);
    expect(screen.getByText(/drag.*drop.*pdf/i)).toBeInTheDocument();
  });

  it("renders upload icon", () => {
    render(<DropZone onFileSelect={mockOnFileSelect} />);
    expect(screen.getByTestId("upload-icon")).toBeInTheDocument();
  });

  it("renders browse button", () => {
    render(<DropZone onFileSelect={mockOnFileSelect} />);
    expect(screen.getByText(/browse/i)).toBeInTheDocument();
  });

  it("accepts PDF files via file input", async () => {
    const user = userEvent.setup();
    render(<DropZone onFileSelect={mockOnFileSelect} />);

    const file = new File(["pdf content"], "test.pdf", { type: "application/pdf" });
    const input = screen.getByTestId("file-input");
    await user.upload(input, file);

    expect(mockOnFileSelect).toHaveBeenCalledWith(file);
  });

  it("rejects non-PDF files via drag and drop", () => {
    render(<DropZone onFileSelect={mockOnFileSelect} />);
    const dropZone = screen.getByTestId("drop-zone");

    const file = new File(["text content"], "test.txt", { type: "text/plain" });
    fireEvent.drop(dropZone, {
      dataTransfer: { files: [file] },
    });

    expect(mockOnFileSelect).not.toHaveBeenCalled();
    expect(screen.getByText(/only pdf files are accepted/i)).toBeInTheDocument();
  });

  it("rejects files exceeding 50MB via drag and drop", () => {
    render(<DropZone onFileSelect={mockOnFileSelect} />);
    const dropZone = screen.getByTestId("drop-zone");

    // Create a mock file object with large size
    const file = new File(["x"], "large.pdf", { type: "application/pdf" });
    Object.defineProperty(file, "size", { value: 51 * 1024 * 1024 });

    fireEvent.drop(dropZone, {
      dataTransfer: { files: [file] },
    });

    expect(mockOnFileSelect).not.toHaveBeenCalled();
    expect(screen.getByText(/file exceeds 50mb limit/i)).toBeInTheDocument();
  });

  it("shows drag-over visual feedback", () => {
    render(<DropZone onFileSelect={mockOnFileSelect} />);
    const dropZone = screen.getByTestId("drop-zone");

    fireEvent.dragEnter(dropZone, {
      dataTransfer: { types: ["Files"] },
    });

    expect(dropZone).toHaveClass("border-primary");
  });

  it("removes drag-over feedback on drag leave", () => {
    render(<DropZone onFileSelect={mockOnFileSelect} />);
    const dropZone = screen.getByTestId("drop-zone");

    fireEvent.dragEnter(dropZone, {
      dataTransfer: { types: ["Files"] },
    });
    fireEvent.dragLeave(dropZone);

    expect(dropZone).not.toHaveClass("border-primary");
  });

  it("accepts PDF via drag and drop", () => {
    render(<DropZone onFileSelect={mockOnFileSelect} />);
    const dropZone = screen.getByTestId("drop-zone");

    const file = new File(["pdf content"], "dropped.pdf", { type: "application/pdf" });
    fireEvent.drop(dropZone, {
      dataTransfer: { files: [file] },
    });

    expect(mockOnFileSelect).toHaveBeenCalledWith(file);
  });

  it("shows selected file name and size", () => {
    render(<DropZone onFileSelect={mockOnFileSelect} selectedFile={new File(["x".repeat(1024)], "paper.pdf", { type: "application/pdf" })} />);
    expect(screen.getByText("paper.pdf")).toBeInTheDocument();
  });

  it("is disabled when disabled prop is true", () => {
    render(<DropZone onFileSelect={mockOnFileSelect} disabled />);
    const dropZone = screen.getByTestId("drop-zone");
    expect(dropZone).toHaveClass("opacity-50");
  });
});
