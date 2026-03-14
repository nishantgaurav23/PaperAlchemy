import { apiClient } from "@/lib/api-client";
import type { UploadResponse } from "@/types/upload";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8002";

/** Raw response shape from backend POST /api/v1/upload */
interface BackendUploadResponse {
  paper_id: string;
  arxiv_id: string;
  title: string;
  authors: string[];
  abstract: string;
  page_count: number;
  chunks_indexed: number;
  parsing_status: string;
  indexing_status: string;
  warnings: string[];
  message: string;
}

/** Backend summary response */
interface BackendSummaryResponse {
  paper_id: string;
  title: string;
  summary: {
    objective: string;
    method: string;
    key_findings: string[];
    contribution: string;
    limitations: string;
  };
  model: string;
  provider: string;
  warnings?: string[];
}

/** Backend highlights response */
interface BackendHighlightsResponse {
  paper_id: string;
  highlights: {
    novel_contributions: string[];
    important_findings: string[];
    practical_implications: string[];
    limitations?: string[];
    keywords?: string[];
  };
}

/** Backend methodology response */
interface BackendMethodologyResponse {
  paper_id: string;
  analysis: {
    research_design: string;
    datasets: Array<{ name: string; description?: string; size?: string }>;
    baselines: string[];
    key_results: Array<{ metric: string; value: string; context?: string }>;
    statistical_significance?: string;
    reproducibility_notes?: string;
  };
}

/**
 * Fetch AI-generated analysis for a paper (summary, highlights, methodology).
 * Returns null for each part that fails (graceful degradation).
 */
async function fetchAnalysis(paperId: string): Promise<{
  summary: BackendSummaryResponse | null;
  highlights: BackendHighlightsResponse | null;
  methodology: BackendMethodologyResponse | null;
}> {
  const [summary, highlights, methodology] = await Promise.allSettled([
    apiClient.post<BackendSummaryResponse>(`/api/v1/papers/${paperId}/summary`, undefined, { timeout: 120000 }),
    apiClient.post<BackendHighlightsResponse>(`/api/v1/papers/${paperId}/highlights`, undefined, { timeout: 120000 }),
    apiClient.post<BackendMethodologyResponse>(`/api/v1/papers/${paperId}/methodology`, undefined, { timeout: 120000 }),
  ]);

  return {
    summary: summary.status === "fulfilled" ? summary.value : null,
    highlights: highlights.status === "fulfilled" ? highlights.value : null,
    methodology: methodology.status === "fulfilled" ? methodology.value : null,
  };
}

function buildUploadResponse(
  raw: BackendUploadResponse,
  analysis: {
    summary: BackendSummaryResponse | null;
    highlights: BackendHighlightsResponse | null;
    methodology: BackendMethodologyResponse | null;
  },
): UploadResponse {
  const s = analysis.summary?.summary;
  const h = analysis.highlights?.highlights;
  const m = analysis.methodology?.analysis;

  return {
    paper: {
      id: raw.paper_id,
      title: raw.title,
      authors: raw.authors,
      abstract: raw.abstract,
      categories: [],
      arxiv_id: raw.arxiv_id,
    },
    summary: s
      ? {
          objective: s.objective,
          method: s.method,
          key_findings: s.key_findings,
          contribution: s.contribution,
          limitations: s.limitations,
        }
      : {
          objective: raw.abstract
            ? `${raw.abstract.slice(0, 500)}${raw.abstract.length > 500 ? "..." : ""}`
            : "No abstract available.",
          method: "AI analysis unavailable — showing abstract as fallback.",
          key_findings: "Upload was successful. Try refreshing the page to trigger analysis.",
          contribution: raw.message,
          limitations: raw.warnings.length > 0 ? raw.warnings.join(". ") : "None detected.",
        },
    highlights: h
      ? {
          novel_contributions: h.novel_contributions,
          important_findings: h.important_findings,
          practical_implications: h.practical_implications,
        }
      : {
          novel_contributions: ["AI highlight extraction unavailable — paper was uploaded successfully."],
          important_findings: ["Use the Chat feature to ask questions about this paper."],
          practical_implications: ["Paper is searchable via hybrid search (BM25 + vector)."],
        },
    methodology: m
      ? {
          approach: m.research_design,
          datasets: m.datasets.map((d) => `${d.name}${d.description ? ` — ${d.description}` : ""}${d.size ? ` (${d.size})` : ""}`),
          baselines: m.baselines,
          results: m.key_results.map((r) => `${r.metric}: ${r.value}${r.context ? ` (${r.context})` : ""}`).join(" | "),
          statistical_significance: m.statistical_significance ?? undefined,
        }
      : {
          approach: "AI methodology analysis unavailable — paper was uploaded successfully.",
          datasets: [],
          baselines: [],
          results: "",
        },
  };
}

/** Mock response for testing and development */
export const MOCK_RESPONSE: UploadResponse = {
  paper: {
    id: "mock-paper-id",
    title: "Attention Is All You Need",
    authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
    abstract:
      "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
    categories: ["cs.CL", "cs.AI"],
    arxiv_id: "1706.03762",
  },
  summary: {
    objective:
      "We propose a new sequence-to-sequence architecture based entirely on attention mechanisms, dispensing with recurrence and convolution.",
    method:
      "The Transformer uses multi-head self-attention in both encoder and decoder, with positional encodings to capture sequence order.",
    key_findings: [
      "Achieves 28.4 BLEU on WMT 2014 English-to-German, surpassing all previous models",
      "Training time reduced to 3.5 days on 8 GPUs",
      "Generalizes well to other tasks like English constituency parsing",
    ],
    contribution:
      "Introduces the Transformer, the first model to rely entirely on self-attention for sequence transduction.",
    limitations:
      "Attention complexity is quadratic in sequence length, which can be prohibitive for very long sequences.",
  },
  highlights: {
    novel_contributions: [
      "First model to rely entirely on self-attention for sequence-to-sequence tasks",
      "Multi-head attention mechanism allows the model to jointly attend to information from different representation subspaces",
    ],
    important_findings: [
      "Outperforms all previously published models on WMT 2014 EN-DE translation",
      "Significantly faster to train than architectures based on recurrent or convolutional layers",
    ],
    practical_implications: [
      "Enables massive parallelization during training",
      "Foundation architecture for subsequent models like BERT and GPT",
    ],
  },
  methodology: {
    approach:
      "Encoder-decoder architecture using stacked self-attention and point-wise, fully connected layers.",
    datasets: ["WMT 2014 English-German (4.5M sentence pairs)", "WMT 2014 English-French (36M sentence pairs)"],
    baselines: ["ByteNet", "ConvS2S", "MoE", "Deep-Att + PosUnk", "GNMT + RL"],
    results: "BLEU: 28.4 (EN-DE) | BLEU: 41.0 (EN-FR) | Training: 3.5 days on 8 P100 GPUs",
  },
};

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${BASE_URL}/api/v1/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorBody: string;
    try {
      const json = await response.json();
      errorBody = json.detail ?? json.error?.message ?? JSON.stringify(json);
    } catch {
      errorBody = await response.text();
    }
    throw new Error(`Upload failed: ${errorBody}`);
  }

  const raw = (await response.json()) as BackendUploadResponse;

  // Fetch real AI analysis from the backend (summary, highlights, methodology)
  const analysis = await fetchAnalysis(raw.paper_id);

  return buildUploadResponse(raw, analysis);
}
