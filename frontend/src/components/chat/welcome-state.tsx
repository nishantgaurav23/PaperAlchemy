"use client";

import { BookOpen, Sparkles } from "lucide-react";
import { SUGGESTED_QUESTIONS } from "@/types/chat";

interface WelcomeStateProps {
  onSelectQuestion: (question: string) => void;
}

export function WelcomeState({ onSelectQuestion }: WelcomeStateProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-8 px-4 py-16">
      {/* Gradient illustration */}
      <div className="relative flex flex-col items-center gap-4 text-center">
        <div className="relative">
          <div className="absolute -inset-4 rounded-full bg-gradient-to-br from-primary/20 via-accent/10 to-primary/5 blur-xl" />
          <div
            data-testid="welcome-icon"
            className="relative flex size-20 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-accent shadow-lg"
          >
            <BookOpen className="size-10 text-white" />
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <Sparkles className="size-4 text-primary" />
          <span className="text-xs font-medium tracking-wider text-primary uppercase">
            AI-Powered Research
          </span>
        </div>
        <h2 className="text-2xl font-bold tracking-tight">
          PaperAlchemy Research Assistant
        </h2>
        <p className="max-w-md text-sm text-muted-foreground">
          Ask me anything about research papers in the knowledge base. I&apos;ll
          search for relevant papers and cite my sources.
        </p>
      </div>

      {/* Example prompts */}
      <div className="flex flex-col gap-3">
        <p className="text-center text-xs font-medium text-muted-foreground">
          Try asking:
        </p>
        <div className="flex flex-col gap-2 sm:grid sm:grid-cols-2">
          {SUGGESTED_QUESTIONS.map((question) => (
            <button
              key={question}
              onClick={() => onSelectQuestion(question)}
              data-testid="suggested-question"
              className="rounded-xl border border-border bg-background/50 px-4 py-3 text-left text-sm text-muted-foreground transition-all hover:border-primary/40 hover:bg-primary/5 hover:text-foreground hover:shadow-sm"
            >
              {question}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
