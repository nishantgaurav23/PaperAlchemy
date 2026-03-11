import { apiClient } from "@/lib/api-client";
import type { PaperDetail, RelatedPapersResponse } from "@/types/paper";

const MOCK_PAPER: PaperDetail = {
  id: "mock-paper-001",
  arxiv_id: "1706.03762",
  title: "Attention Is All You Need",
  authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
  abstract:
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
  categories: ["cs.CL", "cs.AI"],
  published_date: "2017-06-12",
  pdf_url: "https://arxiv.org/pdf/1706.03762",
  sections: [
    { title: "Introduction", content: "Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence." },
    { title: "Methods", content: "The Transformer follows an encoder-decoder structure using stacked self-attention and point-wise, fully connected layers." },
  ],
  summary: {
    objective: "Propose a new network architecture based solely on attention mechanisms",
    method: "Multi-head self-attention with positional encoding",
    key_findings: "Achieves state-of-the-art BLEU scores on WMT 2014 translation tasks",
    contribution: "Introduced the Transformer architecture",
    limitations: "Quadratic complexity with respect to sequence length",
  },
  highlights: {
    novel_contributions: ["Self-attention mechanism replacing recurrence", "Multi-head attention"],
    important_findings: ["28.4 BLEU on EN-DE translation", "41.8 BLEU on EN-FR translation"],
    practical_implications: ["Parallelizable training", "Foundation for BERT, GPT"],
  },
  methodology: {
    approach: "Encoder-decoder with multi-head self-attention",
    datasets: ["WMT 2014 English-German", "WMT 2014 English-French"],
    baselines: ["ConvS2S", "ByteNet", "GNMT"],
    results: "28.4 BLEU on EN-DE, 41.8 BLEU on EN-FR",
  },
};

const MOCK_RELATED: RelatedPapersResponse = {
  papers: [
    {
      id: "mock-related-001",
      arxiv_id: "1810.04805",
      title: "BERT: Pre-training of Deep Bidirectional Transformers",
      authors: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
      abstract: "We introduce a new language representation model called BERT.",
      categories: ["cs.CL"],
      published_date: "2018-10-11",
    },
    {
      id: "mock-related-002",
      arxiv_id: "2005.14165",
      title: "Language Models are Few-Shot Learners",
      authors: ["Tom Brown", "Benjamin Mann"],
      abstract: "We show that scaling up language models greatly improves task-agnostic, few-shot performance.",
      categories: ["cs.CL", "cs.AI"],
      published_date: "2020-05-28",
    },
  ],
};

export async function getPaper(id: string, useMock = false): Promise<PaperDetail> {
  if (useMock) {
    return { ...MOCK_PAPER, id };
  }
  return apiClient.get<PaperDetail>(`/api/v1/papers/${id}`);
}

export async function getRelatedPapers(id: string, useMock = false): Promise<RelatedPapersResponse> {
  if (useMock) {
    return { ...MOCK_RELATED };
  }
  return apiClient.get<RelatedPapersResponse>(`/api/v1/papers/${id}/related`);
}
