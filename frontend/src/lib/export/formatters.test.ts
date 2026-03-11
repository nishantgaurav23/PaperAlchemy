import { describe, it, expect } from "vitest";
import {
  formatBibtex,
  formatMarkdown,
  formatSlideSnippet,
  formatBulkBibtex,
  formatBulkMarkdown,
} from "./formatters";
import type { Paper } from "@/types/paper";

const MOCK_PAPER: Paper = {
  id: "abc-123",
  arxiv_id: "1706.03762",
  title: "Attention Is All You Need",
  authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
  abstract:
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture based solely on attention mechanisms.",
  categories: ["cs.CL", "cs.LG"],
  published_date: "2017-06-12",
};

const MOCK_PAPER_MINIMAL: Paper = {
  id: "def-456",
  arxiv_id: "",
  title: "A Paper With No ArXiv ID",
  authors: [],
  abstract: "",
  categories: [],
  published_date: "",
};

describe("formatBibtex", () => {
  it("generates valid BibTeX for a complete paper", () => {
    const result = formatBibtex(MOCK_PAPER);
    expect(result).toContain("@article{vaswani2017attention");
    expect(result).toContain("author = {Ashish Vaswani and Noam Shazeer and Niki Parmar}");
    expect(result).toContain('title = {Attention Is All You Need}');
    expect(result).toContain("year = {2017}");
    expect(result).toContain("eprint = {1706.03762}");
    expect(result).toContain("archivePrefix = {arXiv}");
    expect(result).toContain("url = {https://arxiv.org/abs/1706.03762}");
  });

  it("escapes special characters in title", () => {
    const paper: Paper = {
      ...MOCK_PAPER,
      title: "BERT & Friends: 100% Better {Results}",
    };
    const result = formatBibtex(paper);
    expect(result).toContain("BERT \\& Friends: 100\\% Better \\{Results\\}");
  });

  it("handles missing authors", () => {
    const result = formatBibtex(MOCK_PAPER_MINIMAL);
    expect(result).toContain("author = {Unknown}");
  });

  it("handles missing arxiv_id", () => {
    const result = formatBibtex(MOCK_PAPER_MINIMAL);
    expect(result).not.toContain("eprint");
    expect(result).not.toContain("archivePrefix");
    expect(result).not.toContain("url = {https://arxiv.org/abs/}");
  });

  it("handles missing published_date", () => {
    const result = formatBibtex(MOCK_PAPER_MINIMAL);
    expect(result).not.toContain("year = {}");
  });

  it("generates a cite key from first author last name and year", () => {
    const result = formatBibtex(MOCK_PAPER);
    expect(result).toMatch(/^@article\{vaswani2017/);
  });
});

describe("formatMarkdown", () => {
  it("generates complete Markdown for a paper", () => {
    const result = formatMarkdown(MOCK_PAPER);
    expect(result).toContain("# Attention Is All You Need");
    expect(result).toContain("**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar");
    expect(result).toContain("**Published:** 2017-06-12");
    expect(result).toContain("**arXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)");
    expect(result).toContain("## Abstract");
    expect(result).toContain("The dominant sequence transduction models");
  });

  it("omits abstract section when missing", () => {
    const result = formatMarkdown(MOCK_PAPER_MINIMAL);
    expect(result).not.toContain("## Abstract");
  });

  it("omits arXiv line when no arxiv_id", () => {
    const result = formatMarkdown(MOCK_PAPER_MINIMAL);
    expect(result).not.toContain("**arXiv:**");
  });

  it("shows categories when present", () => {
    const result = formatMarkdown(MOCK_PAPER);
    expect(result).toContain("**Categories:** cs.CL, cs.LG");
  });
});

describe("formatSlideSnippet", () => {
  it("generates a concise slide snippet", () => {
    const result = formatSlideSnippet(MOCK_PAPER);
    expect(result).toContain("Attention Is All You Need");
    expect(result).toContain("Vaswani, Shazeer, Parmar");
    expect(result).toContain("https://arxiv.org/abs/1706.03762");
  });

  it("truncates authors to 3 + et al. when more than 3", () => {
    const paper: Paper = {
      ...MOCK_PAPER,
      authors: ["Author A", "Author B", "Author C", "Author D", "Author E"],
    };
    const result = formatSlideSnippet(paper);
    expect(result).toContain("A, B, C et al.");
  });

  it("does not add et al. for 3 or fewer authors", () => {
    const result = formatSlideSnippet(MOCK_PAPER);
    expect(result).not.toContain("et al.");
  });

  it("uses first sentence of abstract as key point", () => {
    const result = formatSlideSnippet(MOCK_PAPER);
    expect(result).toContain(
      "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks."
    );
  });

  it("handles single author", () => {
    const paper: Paper = { ...MOCK_PAPER, authors: ["Jane Doe"] };
    const result = formatSlideSnippet(paper);
    expect(result).toContain("Doe");
    expect(result).not.toContain("et al.");
  });

  it("handles no abstract", () => {
    const paper: Paper = { ...MOCK_PAPER, abstract: "" };
    const result = formatSlideSnippet(paper);
    expect(result).not.toContain("Key Point:");
  });
});

describe("formatBulkBibtex", () => {
  it("combines multiple papers into a single BibTeX string", () => {
    const papers = [MOCK_PAPER, { ...MOCK_PAPER, id: "xyz-789", arxiv_id: "1810.04805", title: "BERT" }];
    const result = formatBulkBibtex(papers);
    expect(result).toContain("@article{vaswani2017");
    expect(result).toContain("title = {BERT}");
    // Entries separated by blank line
    const entries = result.split("\n\n").filter((s) => s.trim().startsWith("@article"));
    expect(entries).toHaveLength(2);
  });

  it("returns empty string for empty array", () => {
    expect(formatBulkBibtex([])).toBe("");
  });
});

describe("formatBulkMarkdown", () => {
  it("combines multiple papers into a Markdown list", () => {
    const papers = [MOCK_PAPER, { ...MOCK_PAPER, id: "xyz-789", title: "BERT" }];
    const result = formatBulkMarkdown(papers);
    expect(result).toContain("# Attention Is All You Need");
    expect(result).toContain("# BERT");
  });

  it("returns empty string for empty array", () => {
    expect(formatBulkMarkdown([])).toBe("");
  });
});
