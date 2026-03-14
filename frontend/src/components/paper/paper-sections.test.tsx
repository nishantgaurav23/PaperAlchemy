import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { PaperSections } from "./paper-sections";
import type { PaperSection } from "@/types/paper";

// Content must be > 50 chars to pass the length filter
const mockSections: PaperSection[] = [
  { title: "Introduction", content: "Self-attention mechanisms have shown great promise in various natural language processing tasks and beyond." },
  { title: "Methods", content: "We propose a multi-head attention mechanism that enables the model to attend to information from different representation subspaces." },
  { title: "Results", content: "Our model achieves state-of-the-art results on multiple benchmark datasets, outperforming all previous approaches significantly." },
  { title: "Conclusion", content: "We have presented the Transformer architecture, which relies entirely on attention mechanisms and achieves remarkable results." },
];

describe("PaperSections", () => {
  it("renders section headings", () => {
    render(<PaperSections sections={mockSections} />);
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(screen.getByText("Methods")).toBeInTheDocument();
  });

  it("shows first 3 important sections expanded by default", () => {
    render(<PaperSections sections={mockSections} />);
    expect(
      screen.getByText(/Self-attention mechanisms have shown great promise/)
    ).toBeVisible();
    expect(
      screen.getByText(/We propose a multi-head attention mechanism/)
    ).toBeVisible();
  });

  it("toggles section expansion on click", () => {
    render(<PaperSections sections={mockSections} />);
    // Click on "Introduction" heading to collapse it
    const introButton = screen.getByRole("button", { name: /Introduction/ });
    fireEvent.click(introButton);
    const introContent = screen.getByText(/Self-attention mechanisms have shown great promise/);
    expect(introContent.closest("[data-state]")).toHaveAttribute("data-state", "collapsed");
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
    render(<PaperSections sections={[{ title: "Introduction", content: "Self-attention mechanisms have shown great promise in various natural language processing tasks and beyond." }]} />);
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(
      screen.getByText(/Self-attention mechanisms have shown great promise/)
    ).toBeVisible();
  });

  it("renders Paper Sections heading", () => {
    render(<PaperSections sections={mockSections} />);
    expect(screen.getByText("Paper Sections")).toBeInTheDocument();
  });
});
