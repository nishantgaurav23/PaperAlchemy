import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import { AnalysisResults } from "./analysis-results";
import { MOCK_RESPONSE } from "@/lib/api/upload";

describe("AnalysisResults", () => {
  it("renders paper title and authors", () => {
    render(<AnalysisResults data={MOCK_RESPONSE} onUploadAnother={vi.fn()} />);
    expect(screen.getByText("Attention Is All You Need")).toBeInTheDocument();
    expect(screen.getByText(/Ashish Vaswani/)).toBeInTheDocument();
  });

  it("renders arXiv link when arxiv_id is present", () => {
    render(<AnalysisResults data={MOCK_RESPONSE} onUploadAnother={vi.fn()} />);
    const link = screen.getByRole("link", { name: /arxiv/i });
    expect(link).toHaveAttribute("href", "https://arxiv.org/abs/1706.03762");
    expect(link).toHaveAttribute("target", "_blank");
  });

  it("renders summary tab by default", () => {
    render(<AnalysisResults data={MOCK_RESPONSE} onUploadAnother={vi.fn()} />);
    expect(screen.getByText(/objective/i)).toBeInTheDocument();
    expect(screen.getByText(/propose a new sequence/i)).toBeInTheDocument();
  });

  it("switches to highlights tab", async () => {
    const user = userEvent.setup();
    render(<AnalysisResults data={MOCK_RESPONSE} onUploadAnother={vi.fn()} />);

    await user.click(screen.getByRole("tab", { name: /highlights/i }));
    expect(screen.getByText(/novel contributions/i)).toBeInTheDocument();
    expect(screen.getByText(/first model to rely entirely/i)).toBeInTheDocument();
  });

  it("switches to methodology tab", async () => {
    const user = userEvent.setup();
    render(<AnalysisResults data={MOCK_RESPONSE} onUploadAnother={vi.fn()} />);

    await user.click(screen.getByRole("tab", { name: /methodology/i }));
    expect(screen.getByText(/approach/i)).toBeInTheDocument();
    expect(screen.getByText(/encoder-decoder/i)).toBeInTheDocument();
  });

  it("renders Upload Another button", () => {
    render(<AnalysisResults data={MOCK_RESPONSE} onUploadAnother={vi.fn()} />);
    expect(screen.getByRole("button", { name: /upload another/i })).toBeInTheDocument();
  });

  it("calls onUploadAnother when button is clicked", async () => {
    const user = userEvent.setup();
    const mockOnUploadAnother = vi.fn();
    render(<AnalysisResults data={MOCK_RESPONSE} onUploadAnother={mockOnUploadAnother} />);

    await user.click(screen.getByRole("button", { name: /upload another/i }));
    expect(mockOnUploadAnother).toHaveBeenCalled();
  });

  it("renders categories as badges", () => {
    render(<AnalysisResults data={MOCK_RESPONSE} onUploadAnother={vi.fn()} />);
    expect(screen.getByText("cs.CL")).toBeInTheDocument();
    expect(screen.getByText("cs.AI")).toBeInTheDocument();
  });
});
