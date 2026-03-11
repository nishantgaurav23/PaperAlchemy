import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { PaperSections } from "./paper-sections";
import type { PaperSection } from "@/types/paper";

const mockSections: PaperSection[] = [
  { title: "Introduction", content: "Self-attention mechanisms have shown great promise." },
  { title: "Methods", content: "We propose a multi-head attention mechanism." },
  { title: "Results", content: "Our model achieves state-of-the-art results." },
  { title: "Conclusion", content: "We have presented the Transformer architecture." },
];

describe("PaperSections", () => {
  it("renders section headings", () => {
    render(<PaperSections sections={mockSections} />);
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(screen.getByText("Methods")).toBeInTheDocument();
    expect(screen.getByText("Results")).toBeInTheDocument();
    expect(screen.getByText("Conclusion")).toBeInTheDocument();
  });

  it("shows first 2 sections expanded by default", () => {
    render(<PaperSections sections={mockSections} />);
    expect(
      screen.getByText("Self-attention mechanisms have shown great promise.")
    ).toBeVisible();
    expect(
      screen.getByText("We propose a multi-head attention mechanism.")
    ).toBeVisible();
  });

  it("collapses sections beyond the first 2 by default", () => {
    render(<PaperSections sections={mockSections} />);
    // The 3rd and 4th sections should be collapsed (content hidden)
    const resultContent = screen.getByText("Our model achieves state-of-the-art results.");
    expect(resultContent.closest("[data-state]")).toHaveAttribute("data-state", "collapsed");
  });

  it("toggles section expansion on click", () => {
    render(<PaperSections sections={mockSections} />);
    // Click on "Results" heading to expand it
    const resultsButton = screen.getByRole("button", { name: /Results/ });
    fireEvent.click(resultsButton);
    const resultContent = screen.getByText("Our model achieves state-of-the-art results.");
    expect(resultContent.closest("[data-state]")).toHaveAttribute("data-state", "expanded");
  });

  it("shows fallback message when no sections", () => {
    render(<PaperSections sections={[]} />);
    expect(
      screen.getByText(/sections not yet parsed/i)
    ).toBeInTheDocument();
  });

  it("shows fallback message when sections is undefined", () => {
    render(<PaperSections sections={undefined} />);
    expect(
      screen.getByText(/sections not yet parsed/i)
    ).toBeInTheDocument();
  });

  it("renders single section expanded", () => {
    render(<PaperSections sections={[mockSections[0]]} />);
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(
      screen.getByText("Self-attention mechanisms have shown great promise.")
    ).toBeVisible();
  });
});
