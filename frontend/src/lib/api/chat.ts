import type { ChatSource, ChatStreamEvent } from "@/types/chat";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const MOCK_RESPONSES = [
  {
    text: `Based on the literature, transformer architectures have seen significant advances in efficiency and scalability [1]. Recent work focuses on sparse attention mechanisms that reduce the quadratic complexity of standard self-attention [2]. Additionally, mixture-of-experts models have shown promising results in scaling model capacity without proportionally increasing computation [1].

**Key developments include:**
- Linear attention variants that approximate softmax attention
- Flash Attention for memory-efficient training
- Retrieval-augmented generation combining parametric and non-parametric knowledge

These advances collectively enable training and deployment of larger, more capable models while managing computational costs [2].`,
    sources: [
      {
        title: "Efficient Transformers: A Survey",
        authors: ["Yi Tay", "Mostafa Dehghani", "Dara Bahri", "Donald Metzler"],
        year: 2022,
        arxiv_id: "2009.06732",
      },
      {
        title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        authors: ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Re"],
        year: 2022,
        arxiv_id: "2205.14135",
      },
    ],
  },
  {
    text: `The attention mechanism, introduced in the seminal Transformer paper [1], enables models to weigh the importance of different parts of the input sequence dynamically. The key contributions include:

1. **Self-attention**: Allows each position to attend to all other positions in a single step [1]
2. **Multi-head attention**: Enables the model to jointly attend to information from different representation subspaces [1]
3. **Positional encodings**: Inject sequence order information without recurrence [2]

The mechanism has since been adapted for computer vision (Vision Transformers) [2] and multi-modal learning, fundamentally changing how we process sequential data.`,
    sources: [
      {
        title: "Attention Is All You Need",
        authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year: 2017,
        arxiv_id: "1706.03762",
      },
      {
        title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        authors: ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov"],
        year: 2021,
        arxiv_id: "2010.11929",
      },
    ],
  },
];

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onSources: (sources: ChatSource[]) => void;
  onDone: () => void;
  onError: (error: string) => void;
}

export function streamChat(
  message: string,
  sessionId: string,
  callbacks: StreamCallbacks,
): AbortController {
  const controller = new AbortController();

  if (BASE_URL === "http://localhost:8000" && typeof EventSource === "undefined") {
    // Mock mode: simulate streaming
    streamMock(message, callbacks, controller.signal);
  } else {
    streamReal(message, sessionId, callbacks, controller);
  }

  return controller;
}

function streamMock(
  _message: string,
  callbacks: StreamCallbacks,
  signal: AbortSignal,
): void {
  const response = MOCK_RESPONSES[Math.floor(Math.random() * MOCK_RESPONSES.length)];
  const tokens = response.text.split("");
  let index = 0;

  const interval = setInterval(() => {
    if (signal.aborted) {
      clearInterval(interval);
      return;
    }

    if (index < tokens.length) {
      // Send tokens in small batches for more natural feel
      const batchSize = Math.floor(Math.random() * 3) + 1;
      const batch = tokens.slice(index, index + batchSize).join("");
      callbacks.onToken(batch);
      index += batchSize;
    } else {
      clearInterval(interval);
      callbacks.onSources(response.sources);
      callbacks.onDone();
    }
  }, 20);

  signal.addEventListener("abort", () => {
    clearInterval(interval);
  });
}

function streamReal(
  message: string,
  sessionId: string,
  callbacks: StreamCallbacks,
  controller: AbortController,
): void {
  fetch(`${BASE_URL}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        callbacks.onError(`API Error ${response.status}: ${response.statusText}`);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6);
          if (data === "[DONE]") {
            callbacks.onDone();
            return;
          }

          try {
            const event: ChatStreamEvent = JSON.parse(data);
            switch (event.type) {
              case "token":
                if (event.data) callbacks.onToken(event.data);
                break;
              case "sources":
                if (event.sources) callbacks.onSources(event.sources);
                break;
              case "done":
                callbacks.onDone();
                return;
              case "error":
                callbacks.onError(event.error ?? "Unknown error");
                return;
            }
          } catch {
            // Skip malformed events
          }
        }
      }

      callbacks.onDone();
    })
    .catch((err) => {
      if (err.name === "AbortError") return;
      callbacks.onError(err instanceof Error ? err.message : "Network error");
    });
}
