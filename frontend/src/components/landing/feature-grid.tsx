import {
  MessageSquare,
  Search,
  Upload,
  Quote,
  GitCompareArrows,
  BarChart3,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

interface Feature {
  icon: LucideIcon;
  title: string;
  description: string;
}

const features: Feature[] = [
  {
    icon: MessageSquare,
    title: "AI Chat",
    description:
      "Ask research questions and get citation-backed answers from your paper collection.",
  },
  {
    icon: Search,
    title: "Paper Search",
    description:
      "Hybrid search combining keyword and semantic matching across indexed papers.",
  },
  {
    icon: Upload,
    title: "PDF Upload & Analysis",
    description:
      "Upload papers for AI-generated summaries, highlights, and methodology extraction.",
  },
  {
    icon: Quote,
    title: "Citation-Backed Answers",
    description:
      "Every response includes inline citations with paper titles, authors, and arXiv links.",
  },
  {
    icon: GitCompareArrows,
    title: "Paper Comparison",
    description:
      "Compare papers side-by-side on methodology, findings, and contributions.",
  },
  {
    icon: BarChart3,
    title: "Research Dashboard",
    description:
      "Visualize your research activity with category breakdowns and timeline charts.",
  },
];

export function FeatureGrid() {
  return (
    <section className="px-4 py-20">
      <div className="mx-auto max-w-6xl">
        <h2 className="mb-12 text-center text-3xl font-bold tracking-tight">
          Powerful Features for Researchers
        </h2>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <article
              key={feature.title}
              className="group rounded-xl border bg-card p-6 transition-all duration-200 hover:-translate-y-1 hover:shadow-lg"
            >
              <feature.icon className="mb-4 h-8 w-8 text-primary" />
              <h3 className="mb-2 text-lg font-semibold">{feature.title}</h3>
              <p className="text-sm text-muted-foreground">
                {feature.description}
              </p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
