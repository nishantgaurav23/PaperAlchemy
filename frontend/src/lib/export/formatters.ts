import type { Paper } from "@/types/paper";

function escapeBibtex(text: string): string {
  return text
    .replace(/\\/g, "\\\\")
    .replace(/\{/g, "\\{")
    .replace(/\}/g, "\\}")
    .replace(/&/g, "\\&")
    .replace(/%/g, "\\%");
}

function generateCiteKey(paper: Paper): string {
  const lastNamePart = paper.authors.length > 0
    ? paper.authors[0].split(/\s+/).pop()?.toLowerCase() ?? "unknown"
    : "unknown";
  const year = paper.published_date
    ? new Date(paper.published_date).getFullYear()
    : "";
  const titleWord = paper.title
    .split(/\s+/)
    .find((w) => w.length > 3 && /^[a-zA-Z]/.test(w))
    ?.toLowerCase() ?? "paper";
  return `${lastNamePart}${year}${titleWord}`;
}

export function formatBibtex(paper: Paper): string {
  const key = generateCiteKey(paper);
  const authors =
    paper.authors.length > 0
      ? paper.authors.join(" and ")
      : "Unknown";
  const lines: string[] = [`@article{${key},`];
  lines.push(`  author = {${authors}},`);
  lines.push(`  title = {${escapeBibtex(paper.title)}},`);
  if (paper.published_date) {
    lines.push(`  year = {${new Date(paper.published_date).getFullYear()}},`);
  }
  if (paper.arxiv_id) {
    lines.push(`  eprint = {${paper.arxiv_id}},`);
    lines.push(`  archivePrefix = {arXiv},`);
    if (paper.categories.length > 0) {
      lines.push(`  primaryClass = {${paper.categories[0]}},`);
    }
    lines.push(`  url = {https://arxiv.org/abs/${paper.arxiv_id}},`);
  }
  lines.push("}");
  return lines.join("\n");
}

export function formatMarkdown(paper: Paper): string {
  const lines: string[] = [`# ${paper.title}`];
  lines.push("");
  if (paper.authors.length > 0) {
    lines.push(`**Authors:** ${paper.authors.join(", ")}`);
  }
  if (paper.published_date) {
    lines.push(`**Published:** ${paper.published_date}`);
  }
  if (paper.categories.length > 0) {
    lines.push(`**Categories:** ${paper.categories.join(", ")}`);
  }
  if (paper.arxiv_id) {
    lines.push(`**arXiv:** [${paper.arxiv_id}](https://arxiv.org/abs/${paper.arxiv_id})`);
  }
  if (paper.abstract) {
    lines.push("");
    lines.push("## Abstract");
    lines.push("");
    lines.push(paper.abstract);
  }
  lines.push("");
  return lines.join("\n");
}

function getLastNames(authors: string[]): string[] {
  return authors.map((a) => a.split(/\s+/).pop() ?? a);
}

export function formatSlideSnippet(paper: Paper): string {
  const lines: string[] = [`**${paper.title}**`];
  if (paper.authors.length > 0) {
    const lastNames = getLastNames(paper.authors);
    if (lastNames.length > 3) {
      lines.push(`${lastNames.slice(0, 3).join(", ")} et al.`);
    } else {
      lines.push(lastNames.join(", "));
    }
  }
  if (paper.published_date) {
    lines.push(`(${new Date(paper.published_date).getFullYear()})`);
  }
  if (paper.abstract) {
    const firstSentence = paper.abstract.match(/^[^.!?]+[.!?]/)?.[0] ?? paper.abstract;
    lines.push("");
    lines.push(`Key Point: ${firstSentence}`);
  }
  if (paper.arxiv_id) {
    lines.push("");
    lines.push(`https://arxiv.org/abs/${paper.arxiv_id}`);
  }
  return lines.join("\n");
}

export function formatBulkBibtex(papers: Paper[]): string {
  if (papers.length === 0) return "";
  return papers.map(formatBibtex).join("\n\n");
}

export function formatBulkMarkdown(papers: Paper[]): string {
  if (papers.length === 0) return "";
  return papers.map(formatMarkdown).join("\n---\n\n");
}
