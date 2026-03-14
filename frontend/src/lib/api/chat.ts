import type { ChatSource } from "@/types/chat";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8002";

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
  streamReal(message, sessionId, callbacks, controller);
  return controller;
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
    body: JSON.stringify({ query: message, session_id: sessionId, stream: true }),
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
      let doneEmitted = false;
      // Persist across read() calls so event/data split across chunks works
      let currentEvent: string | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        // Parse SSE named events: "event: <type>\ndata: <json>"
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
            continue;
          }
          if (line.startsWith("data: ") && currentEvent) {
            const data = line.slice(6);
            try {
              const parsed = JSON.parse(data);
              switch (currentEvent) {
                case "metadata":
                  // metadata event — ignore for now (session info)
                  break;
                case "token":
                  if (parsed.text) callbacks.onToken(parsed.text);
                  break;
                case "sources":
                  if (Array.isArray(parsed)) {
                    callbacks.onSources(parsed);
                  }
                  break;
                case "done":
                  callbacks.onDone();
                  doneEmitted = true;
                  return;
                case "error":
                  callbacks.onError(parsed.detail ?? "Unknown error");
                  doneEmitted = true;
                  return;
              }
            } catch {
              // Skip malformed JSON
            }
            continue;
          }
          // Empty line resets event state (SSE spec: dispatches event and resets)
          if (line.trim() === "") {
            currentEvent = null;
          }
        }
      }

      if (!doneEmitted) {
        callbacks.onDone();
      }
    })
    .catch((err) => {
      if (err.name === "AbortError") return;
      callbacks.onError(err instanceof Error ? err.message : "Network error");
    });
}
