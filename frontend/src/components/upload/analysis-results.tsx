"use client";

import { useState } from "react";
import { ExternalLink, Upload, FileText, Lightbulb, FlaskConical } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { UploadResponse } from "@/types/upload";

interface AnalysisResultsProps {
  data: UploadResponse;
  onUploadAnother: () => void;
}

type TabId = "summary" | "highlights" | "methodology";

const TABS: { id: TabId; label: string; icon: typeof FileText }[] = [
  { id: "summary", label: "Summary", icon: FileText },
  { id: "highlights", label: "Highlights", icon: Lightbulb },
  { id: "methodology", label: "Methodology", icon: FlaskConical },
];

export function AnalysisResults({ data, onUploadAnother }: AnalysisResultsProps) {
  const [activeTab, setActiveTab] = useState<TabId>("summary");

  const { paper, summary, highlights, methodology } = data;

  return (
    <div className="flex flex-col gap-6">
      {/* Paper header */}
      <div className="flex flex-col gap-3 rounded-xl border border-border bg-card p-6">
        <h2 className="text-xl font-bold leading-tight">{paper.title}</h2>

        <p className="text-sm text-muted-foreground">
          {paper.authors.length > 3
            ? `${paper.authors.slice(0, 3).join(", ")} et al.`
            : paper.authors.join(", ")}
          {paper.published_date && ` · ${paper.published_date}`}
        </p>

        <div className="flex flex-wrap items-center gap-2">
          {paper.categories.map((cat) => (
            <Badge key={cat} variant="secondary">
              {cat}
            </Badge>
          ))}

          {paper.arxiv_id && (
            <a
              href={`https://arxiv.org/abs/${paper.arxiv_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
            >
              arXiv
              <ExternalLink className="size-3" />
            </a>
          )}
        </div>

        {paper.abstract && (
          <p className="text-sm leading-relaxed text-muted-foreground">{paper.abstract}</p>
        )}
      </div>

      {/* Tabs */}
      <div className="flex flex-col gap-4">
        <div role="tablist" className="flex gap-1 rounded-lg bg-muted p-1">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={activeTab === tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  activeTab === tab.id
                    ? "bg-background text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                <Icon className="size-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        <div className="rounded-xl border border-border bg-card p-6">
          {activeTab === "summary" && <SummaryTab summary={summary} />}
          {activeTab === "highlights" && <HighlightsTab highlights={highlights} />}
          {activeTab === "methodology" && <MethodologyTab methodology={methodology} />}
        </div>
      </div>

      {/* Upload Another */}
      <div className="flex justify-center">
        <Button variant="outline" size="lg" onClick={onUploadAnother}>
          <Upload className="size-4" />
          Upload Another Paper
        </Button>
      </div>
    </div>
  );
}

function SummaryTab({ summary }: { summary: UploadResponse["summary"] }) {
  const sections = [
    { label: "Objective", value: summary.objective },
    { label: "Method", value: summary.method },
    { label: "Key Findings", value: summary.key_findings },
    { label: "Contribution", value: summary.contribution },
    { label: "Limitations", value: summary.limitations },
  ];

  return (
    <div className="flex flex-col gap-4">
      {sections.map(
        (section) =>
          section.value && (
            <div key={section.label}>
              <h3 className="mb-1 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                {section.label}
              </h3>
              {Array.isArray(section.value) ? (
                <ul className="flex flex-col gap-1.5">
                  {section.value.map((item, i) => (
                    <li key={i} className="flex gap-2 text-sm leading-relaxed text-foreground">
                      <span className="mt-1 size-1.5 shrink-0 rounded-full bg-primary" />
                      {item}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm leading-relaxed text-foreground">{section.value}</p>
              )}
            </div>
          ),
      )}
    </div>
  );
}

function HighlightsTab({ highlights }: { highlights: UploadResponse["highlights"] }) {
  const sections = [
    { label: "Novel Contributions", items: highlights.novel_contributions },
    { label: "Important Findings", items: highlights.important_findings },
    { label: "Practical Implications", items: highlights.practical_implications },
  ];

  return (
    <div className="flex flex-col gap-5">
      {sections.map(
        (section) =>
          section.items.length > 0 && (
            <div key={section.label}>
              <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                {section.label}
              </h3>
              <ul className="flex flex-col gap-1.5">
                {section.items.map((item, i) => (
                  <li key={i} className="flex gap-2 text-sm leading-relaxed text-foreground">
                    <span className="mt-1 size-1.5 shrink-0 rounded-full bg-primary" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ),
      )}
    </div>
  );
}

function MethodologyTab({ methodology }: { methodology: UploadResponse["methodology"] }) {
  return (
    <div className="flex flex-col gap-4">
      {methodology.approach && (
        <div>
          <h3 className="mb-1 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Approach
          </h3>
          <p className="text-sm leading-relaxed text-foreground">{methodology.approach}</p>
        </div>
      )}

      {methodology.datasets.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Datasets
          </h3>
          <ul className="flex flex-col gap-1">
            {methodology.datasets.map((ds, i) => (
              <li key={i} className="flex gap-2 text-sm">
                <span className="mt-1 size-1.5 shrink-0 rounded-full bg-primary" />
                {ds}
              </li>
            ))}
          </ul>
        </div>
      )}

      {methodology.baselines.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Baselines
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {methodology.baselines.map((bl) => (
              <Badge key={bl} variant="outline">
                {bl}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {methodology.results && (
        <div>
          <h3 className="mb-1 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Results
          </h3>
          <p className="text-sm leading-relaxed text-foreground">{methodology.results}</p>
        </div>
      )}

      {methodology.statistical_significance && (
        <div>
          <h3 className="mb-1 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Statistical Significance
          </h3>
          <p className="text-sm leading-relaxed">{methodology.statistical_significance}</p>
        </div>
      )}
    </div>
  );
}
