import { apiClient } from "@/lib/api-client";
import type { DashboardData } from "@/types/dashboard";

const MOCK_DASHBOARD_DATA: DashboardData = {
  stats: {
    total_papers: 1247,
    papers_this_week: 42,
    total_categories: 10,
    most_active_category: "cs.AI",
  },
  categories: [
    { category: "cs.AI", count: 324 },
    { category: "cs.CL", count: 289 },
    { category: "cs.LG", count: 256 },
    { category: "cs.CV", count: 198 },
    { category: "cs.IR", count: 87 },
    { category: "cs.NE", count: 45 },
    { category: "stat.ML", count: 32 },
    { category: "cs.RO", count: 16 },
  ],
  timeline: [
    { month: "2024-07", count: 85 },
    { month: "2024-08", count: 102 },
    { month: "2024-09", count: 118 },
    { month: "2024-10", count: 134 },
    { month: "2024-11", count: 155 },
    { month: "2024-12", count: 142 },
    { month: "2025-01", count: 167 },
    { month: "2025-02", count: 178 },
    { month: "2025-03", count: 166 },
  ],
  hot_papers: [
    {
      id: "mock-hot-001",
      arxiv_id: "2501.12345",
      title: "Scaling Laws for Neural Language Models Revisited",
      authors: ["Alice Chen", "Bob Smith", "Carol Davis"],
      abstract: "We revisit scaling laws for neural language models and find new insights on compute-optimal training.",
      categories: ["cs.CL", "cs.AI"],
      published_date: "2025-01-15",
    },
    {
      id: "mock-hot-002",
      arxiv_id: "2502.67890",
      title: "Efficient Transformers: A Survey of Recent Advances",
      authors: ["David Lee", "Emma Wilson"],
      abstract: "A comprehensive survey of efficient transformer architectures including linear attention and sparse methods.",
      categories: ["cs.LG"],
      published_date: "2025-02-20",
    },
    {
      id: "mock-hot-003",
      arxiv_id: "2503.11111",
      title: "Vision-Language Models for Scientific Document Understanding",
      authors: ["Frank Zhang", "Grace Kim", "Henry Park", "Iris Johnson"],
      abstract: "We present a novel vision-language model specifically designed for understanding scientific papers and figures.",
      categories: ["cs.CV", "cs.CL"],
      published_date: "2025-03-05",
    },
  ],
  trending_topics: [
    { topic: "Large Language Models", count: 89 },
    { topic: "Retrieval-Augmented Generation", count: 67 },
    { topic: "Diffusion Models", count: 54 },
    { topic: "Multimodal Learning", count: 48 },
    { topic: "Reinforcement Learning from Human Feedback", count: 42 },
    { topic: "Vision Transformers", count: 38 },
    { topic: "Graph Neural Networks", count: 31 },
    { topic: "Federated Learning", count: 25 },
  ],
};

export async function getDashboardData(useMock = false): Promise<DashboardData> {
  if (useMock) {
    return { ...MOCK_DASHBOARD_DATA };
  }

  try {
    return await apiClient.get<DashboardData>("/api/v1/dashboard/stats");
  } catch {
    return { ...MOCK_DASHBOARD_DATA };
  }
}

export { MOCK_DASHBOARD_DATA };
