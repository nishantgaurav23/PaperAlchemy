export interface UploadedPaper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  categories: string[];
  arxiv_id?: string;
  published_date?: string;
  pdf_url?: string;
}

export interface PaperSummary {
  objective: string;
  method: string;
  key_findings: string;
  contribution: string;
  limitations: string;
}

export interface PaperHighlights {
  novel_contributions: string[];
  important_findings: string[];
  practical_implications: string[];
}

export interface MethodologyAnalysis {
  approach: string;
  datasets: string[];
  baselines: string[];
  results: string;
  statistical_significance?: string;
}

export interface UploadResponse {
  paper: UploadedPaper;
  summary: PaperSummary;
  highlights: PaperHighlights;
  methodology: MethodologyAnalysis;
}

export type UploadStatus = "idle" | "uploading" | "processing" | "complete" | "error";

export const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
export const ACCEPTED_FILE_TYPE = "application/pdf";
