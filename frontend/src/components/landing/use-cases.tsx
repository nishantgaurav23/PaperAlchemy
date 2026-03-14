import { BookOpen, TrendingUp, Microscope } from "lucide-react";
import type { LucideIcon } from "lucide-react";

interface UseCase {
  icon: LucideIcon;
  title: string;
  description: string;
}

const useCases: UseCase[] = [
  {
    icon: BookOpen,
    title: "Literature Review",
    description:
      "Quickly survey a research area by asking questions across hundreds of indexed papers. Get structured summaries with full citation trails.",
  },
  {
    icon: TrendingUp,
    title: "Stay Current",
    description:
      "Keep up with the latest publications in your field. Search by topic, browse recent uploads, and track emerging research trends.",
  },
  {
    icon: Microscope,
    title: "Deep Dive Analysis",
    description:
      "Upload a paper for AI-powered analysis — extract key findings, methodology details, and compare against related work in your collection.",
  },
];

export function UseCases() {
  return (
    <section className="px-4 py-20">
      <div className="mx-auto max-w-5xl">
        <h2 className="mb-12 text-center text-3xl font-bold tracking-tight">
          Built for Real Use Cases
        </h2>

        <div className="grid gap-8 md:grid-cols-3">
          {useCases.map((useCase) => (
            <article
              key={useCase.title}
              className="flex flex-col items-center gap-4 rounded-xl p-6 text-center"
            >
              <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10">
                <useCase.icon className="h-7 w-7 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">{useCase.title}</h3>
              <p className="text-sm text-muted-foreground">
                {useCase.description}
              </p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
