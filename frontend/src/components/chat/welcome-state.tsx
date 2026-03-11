"use client";

import { BookOpen } from "lucide-react";
import { SUGGESTED_QUESTIONS } from "@/types/chat";

interface WelcomeStateProps {
  onSelectQuestion: (question: string) => void;
}

export function WelcomeState({ onSelectQuestion }: WelcomeStateProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-6 px-4 py-16">
      <div className="flex flex-col items-center gap-3 text-center">
        <div className="flex size-16 items-center justify-center rounded-2xl bg-primary/10">
          <BookOpen className="size-8 text-primary" />
        </div>
        <h2 className="text-2xl font-bold tracking-tight">
          PaperAlchemy Research Assistant
        </h2>
        <p className="max-w-md text-sm text-muted-foreground">
          Ask me anything about research papers in the knowledge base. I&apos;ll
          search for relevant papers and cite my sources.
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <p className="text-center text-xs font-medium text-muted-foreground">
          Try asking:
        </p>
        <div className="flex flex-col gap-2 sm:grid sm:grid-cols-2">
          {SUGGESTED_QUESTIONS.map((question) => (
            <button
              key={question}
              onClick={() => onSelectQuestion(question)}
              data-testid="suggested-question"
              className="rounded-lg border border-border px-4 py-3 text-left text-sm text-muted-foreground transition-colors hover:border-primary/50 hover:bg-accent hover:text-foreground"
            >
              {question}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
