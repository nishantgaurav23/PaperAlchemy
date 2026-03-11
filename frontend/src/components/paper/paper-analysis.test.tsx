import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { PaperAnalysis } from "./paper-analysis";
import type { PaperSummary, PaperHighlights, MethodologyAnalysis } from "@/types/upload";

const mockSummary: PaperSummary = {
  objective: "Replace recurrence with attention",
  method: "Multi-head self-attention",
  key_findings: "Achieves SOTA on translation",
  contribution: "Transformer architecture",
  limitations: "Quadratic complexity",
};

const mockHighlights: PaperHighlights = {
  novel_contributions: ["Self-attention mechanism"],
  important_findings: ["BLEU score improvement"],
  practical_implications: ["Parallelizable training"],
};

const mockMethodology: MethodologyAnalysis = {
  approach: "Encoder-decoder with attention",
  datasets: ["WMT 2014"],
  baselines: ["ConvS2S", "ByteNet"],
  results: "28.4 BLEU on EN-DE",
};

describe("PaperAnalysis", () => {
  it("renders summary tab by default", () => {
    render(
      <PaperAnalysis
        summary={mockSummary}
        highlights={mockHighlights}
        methodology={mockMethodology}
      />
    );
    expect(screen.getByText("Replace recurrence with attention")).toBeInTheDocument();
    expect(screen.getByText("Multi-head self-attention")).toBeInTheDocument();
  });

  it("renders all three tab buttons", () => {
    render(
      <PaperAnalysis
        summary={mockSummary}
        highlights={mockHighlights}
        methodology={mockMethodology}
      />
    );
    expect(screen.getByRole("tab", { name: /summary/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /highlights/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /methodology/i })).toBeInTheDocument();
  });

  it("switches to highlights tab on click", () => {
    render(
      <PaperAnalysis
        summary={mockSummary}
        highlights={mockHighlights}
        methodology={mockMethodology}
      />
    );
    fireEvent.click(screen.getByRole("tab", { name: /highlights/i }));
    expect(screen.getByText("Self-attention mechanism")).toBeInTheDocument();
    expect(screen.getByText("BLEU score improvement")).toBeInTheDocument();
  });

  it("switches to methodology tab on click", () => {
    render(
      <PaperAnalysis
        summary={mockSummary}
        highlights={mockHighlights}
        methodology={mockMethodology}
      />
    );
    fireEvent.click(screen.getByRole("tab", { name: /methodology/i }));
    expect(screen.getByText("Encoder-decoder with attention")).toBeInTheDocument();
    expect(screen.getByText("WMT 2014")).toBeInTheDocument();
  });

  it("shows 'Request Analysis' CTA when no analysis data", () => {
    render(<PaperAnalysis />);
    expect(screen.getByText(/analysis not yet available/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /request analysis/i })).toBeInTheDocument();
  });

  it("shows analysis when only summary is available", () => {
    render(<PaperAnalysis summary={mockSummary} />);
    expect(screen.getByText("Replace recurrence with attention")).toBeInTheDocument();
  });
});
