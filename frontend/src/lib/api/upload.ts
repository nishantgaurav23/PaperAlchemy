import type { UploadResponse } from "@/types/upload";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const MOCK_RESPONSE: UploadResponse = {
  paper: {
    id: "mock-paper-001",
    title: "Attention Is All You Need",
    authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
    abstract:
      "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    categories: ["cs.CL", "cs.AI"],
    arxiv_id: "1706.03762",
    published_date: "2017-06-12",
  },
  summary: {
    objective:
      "Propose a new sequence transduction architecture based entirely on attention mechanisms, eliminating the need for recurrence and convolutions.",
    method:
      "Introduced the Transformer model using multi-head self-attention and positional encodings. Trained on WMT 2014 English-to-German and English-to-French translation tasks.",
    key_findings:
      "The Transformer achieves 28.4 BLEU on English-to-German translation, improving over existing best results by over 2 BLEU. Training is significantly more parallelizable and requires less time to train.",
    contribution:
      "Established attention mechanisms as a viable standalone approach for sequence modeling, replacing recurrence entirely. The architecture became the foundation for BERT, GPT, and all modern LLMs.",
    limitations:
      "Quadratic memory complexity with respect to sequence length. Limited to fixed-length context windows. Positional encodings may not generalize well to sequences longer than those seen during training.",
  },
  highlights: {
    novel_contributions: [
      "First model to rely entirely on self-attention for sequence transduction",
      "Multi-head attention allows the model to attend to information from different representation subspaces",
      "Scaled dot-product attention provides efficient computation",
    ],
    important_findings: [
      "Achieves new state-of-the-art BLEU scores on WMT 2014 translation benchmarks",
      "Training time reduced to a fraction of competing approaches",
      "Generalizes well to English constituency parsing",
    ],
    practical_implications: [
      "Enables highly parallelized training on modern GPU hardware",
      "Foundation architecture adopted across NLP, vision, and multi-modal AI",
      "Simplified architecture makes implementation and modification easier",
    ],
  },
  methodology: {
    approach:
      "Encoder-decoder architecture with stacked self-attention and point-wise fully connected layers. Uses multi-head attention with 8 heads and 512-dimensional model.",
    datasets: ["WMT 2014 English-German (4.5M sentence pairs)", "WMT 2014 English-French (36M sentence pairs)"],
    baselines: ["ByteNet", "Deep-Att + PosUnk", "GNMT + RL", "ConvS2S", "MoE"],
    results:
      "28.4 BLEU on EN-DE (new SOTA, +2.0 over previous best). 41.0 BLEU on EN-FR (new SOTA). Base model trained in 12 hours on 8 P100 GPUs.",
    statistical_significance: "Results consistently outperform baselines across multiple random seeds and evaluation sets.",
  },
};

export async function uploadPdf(file: File): Promise<UploadResponse> {
  // Mock mode for development
  if (typeof window !== "undefined" && BASE_URL === "http://localhost:8000") {
    return uploadMock();
  }
  return uploadReal(file);
}

async function uploadMock(): Promise<UploadResponse> {
  // Simulate upload + processing delay
  await new Promise((resolve) => setTimeout(resolve, 2000));
  return MOCK_RESPONSE;
}

async function uploadReal(file: File): Promise<UploadResponse> {
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
      errorBody = json.detail ?? JSON.stringify(json);
    } catch {
      errorBody = await response.text();
    }
    throw new Error(`Upload failed: ${errorBody}`);
  }

  return (await response.json()) as UploadResponse;
}

export { MOCK_RESPONSE };
