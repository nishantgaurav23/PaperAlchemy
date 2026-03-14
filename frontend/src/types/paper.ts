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
  parsing_status?: string;
  pdf_content?: string;
  summary?: {
    objective: string;
    method: string;
    key_findings: string[];
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
    datasets: Array<{ name: string; description?: string; size?: string }> | string[];
    baselines: string[];
    key_results: Array<{ metric: string; value: string; comparison?: string }> | string[];
    results?: string; // legacy compat
    statistical_significance?: string;
  };
}

export interface RelatedPapersResponse {
  papers: Paper[];
}

/** Mirrors backend SearchHit from POST /api/v1/search */
export interface SearchHit {
  arxiv_id: string;
  title: string;
  authors: string[];
  abstract: string;
  pdf_url: string;
  score: number;
  highlights: Record<string, unknown>;
  chunk_text: string;
  chunk_id: string;
  section_title?: string | null;
}

/** Mirrors backend SearchResponse from POST /api/v1/search */
export interface SearchResponse {
  query: string;
  total: number;
  hits: SearchHit[];
  size: number;
  from: number;
  search_mode: string;
}

/** Parameters the frontend search UI collects */
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

/** Mirrors backend ArxivSearchHit from POST /api/v1/search/arxiv */
export interface ArxivSearchHit {
  arxiv_id: string;
  title: string;
  authors: string[];
  abstract: string;
  categories: string[];
  published_date: string;
  pdf_url: string;
  arxiv_url: string;
}

/** Mirrors backend ArxivSearchResponse from POST /api/v1/search/arxiv */
export interface ArxivSearchResponse {
  query: string;
  total: number;
  hits: ArxivSearchHit[];
}
