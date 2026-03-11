export interface Paper {
  id: string;
  arxiv_id: string;
  title: string;
  authors: string[];
  abstract: string;
  categories: string[];
  published_date: string;
  pdf_url?: string;
}

export interface PaperSection {
  title: string;
  content: string;
}

export interface PaperDetail extends Paper {
  sections?: PaperSection[];
  summary?: {
    objective: string;
    method: string;
    key_findings: string;
    contribution: string;
    limitations: string;
  };
  highlights?: {
    novel_contributions: string[];
    important_findings: string[];
    practical_implications: string[];
  };
  methodology?: {
    approach: string;
    datasets: string[];
    baselines: string[];
    results: string;
    statistical_significance?: string;
  };
}

export interface RelatedPapersResponse {
  papers: Paper[];
}

export interface SearchResponse {
  papers: Paper[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface SearchParams {
  q?: string;
  category?: string;
  sort?: string;
  page?: number;
}

export const ARXIV_CATEGORIES = [
  { value: "cs.AI", label: "Artificial Intelligence" },
  { value: "cs.CL", label: "Computation and Language" },
  { value: "cs.CV", label: "Computer Vision" },
  { value: "cs.LG", label: "Machine Learning" },
  { value: "cs.NE", label: "Neural & Evolutionary" },
  { value: "cs.IR", label: "Information Retrieval" },
  { value: "cs.RO", label: "Robotics" },
  { value: "stat.ML", label: "Statistics - ML" },
  { value: "math.OC", label: "Optimization & Control" },
  { value: "eess.SP", label: "Signal Processing" },
] as const;

export const SORT_OPTIONS = [
  { value: "relevance", label: "Relevance" },
  { value: "date_desc", label: "Newest First" },
  { value: "date_asc", label: "Oldest First" },
] as const;

export const PAGE_SIZE = 20;
