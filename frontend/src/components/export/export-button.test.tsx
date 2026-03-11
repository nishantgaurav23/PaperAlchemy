import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ExportButton } from "./export-button";
import * as clipboard from "@/lib/export/clipboard";
import type { Paper } from "@/types/paper";

vi.mock("@/lib/export/clipboard", () => ({
  copyToClipboard: vi.fn().mockResolvedValue(true),
  downloadFile: vi.fn(),
}));

const MOCK_PAPER: Paper = {
  id: "abc-123",
  arxiv_id: "1706.03762",
  title: "Attention Is All You Need",
  authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
  abstract: "The dominant sequence transduction models.",
  categories: ["cs.CL"],
  published_date: "2017-06-12",
};

describe("ExportButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the export button", () => {
    render(<ExportButton papers={[MOCK_PAPER]} />);
    expect(screen.getByRole("button", { name: /export/i })).toBeInTheDocument();
  });

  it("shows format options when clicked", () => {
    render(<ExportButton papers={[MOCK_PAPER]} />);
    fireEvent.click(screen.getByRole("button", { name: /export/i }));
    expect(screen.getByText("BibTeX")).toBeInTheDocument();
    expect(screen.getByText("Markdown")).toBeInTheDocument();
    expect(screen.getByText("Slide Snippet")).toBeInTheDocument();
  });

  it("copies BibTeX to clipboard when copy action clicked", async () => {
    render(<ExportButton papers={[MOCK_PAPER]} />);
    fireEvent.click(screen.getByRole("button", { name: /export/i }));

    const bibtexItems = screen.getAllByText("BibTeX");
    // Find the menu item (not the heading)
    const copyBibtex = bibtexItems.find(
      (el) => el.closest("[data-action='copy']") !== null
    ) ?? bibtexItems[0];
    fireEvent.click(copyBibtex);

    await waitFor(() => {
      expect(clipboard.copyToClipboard).toHaveBeenCalled();
    });
  });

  it("downloads BibTeX file when download action clicked", () => {
    render(<ExportButton papers={[MOCK_PAPER]} />);
    fireEvent.click(screen.getByRole("button", { name: /export/i }));

    const downloadButtons = screen.getAllByLabelText(/download bibtex/i);
    fireEvent.click(downloadButtons[0]);

    expect(clipboard.downloadFile).toHaveBeenCalled();
    const [, filename] = (clipboard.downloadFile as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(filename).toMatch(/\.bib$/);
  });

  it("handles bulk export label for multiple papers", () => {
    render(<ExportButton papers={[MOCK_PAPER, MOCK_PAPER]} label="Export All" />);
    expect(screen.getByRole("button", { name: /export all/i })).toBeInTheDocument();
  });
});
