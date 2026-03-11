export interface ChatSource {
  title: string;
  authors: string[];
  year: number;
  arxiv_id: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  sources?: ChatSource[];
  timestamp: number;
}

export interface ChatStreamEvent {
  type: "token" | "sources" | "done" | "error";
  data?: string;
  sources?: ChatSource[];
  error?: string;
}

export const SUGGESTED_QUESTIONS = [
  "What are the latest advances in transformer architectures?",
  "Explain the key contributions of attention mechanisms",
  "Compare BERT and GPT approaches to NLP",
  "What methods are used for efficient fine-tuning of LLMs?",
] as const;

export const MAX_MESSAGE_LENGTH = 2000;
