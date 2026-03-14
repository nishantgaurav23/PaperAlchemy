"use client";

import { useState } from "react";
import { FileText, Lightbulb, FlaskConical, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { PaperSummary, PaperHighlights, MethodologyAnalysis } from "@/types/upload";

interface PaperAnalysisProps {
  summary?: PaperSummary;
  highlights?: PaperHighlights;
  methodology?: MethodologyAnalysis;
  onRequestAnalysis?: () => void;
  isAnalyzing?: boolean;
}

type TabId = "summary" | "highlights" | "methodology";

const TABS: { id: TabId; label: string; icon: typeof FileText }[] = [
  { id: "summary", label: "Summary", icon: FileText },
  { id: "highlights", label: "Highlights", icon: Lightbulb },
  { id: "methodology", label: "Methodology", icon: FlaskConical },
];

export function PaperAnalysis({
  summary,
  highlights,
  methodology,
  onRequestAnalysis,
  isAnalyzing = false,
}: PaperAnalysisProps) {
  const [activeTab, setActiveTab] = useState<TabId>("summary");

  const hasAnalysis = summary || highlights || methodology;

  if (!hasAnalysis) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-lg border border-dashed border-border p-8 text-center">
        <Sparkles className="size-8 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">
          {isAnalyzing
            ? "Generating AI analysis... This may take a moment."
            : "Analysis not yet available for this paper."}
        </p>
        <Button variant="outline" onClick={onRequestAnalysis} disabled={isAnalyzing}>
          {isAnalyzing ? "Analyzing..." : "Request Analysis"}
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-lg font-semibold">Analysis</h2>

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

      <div className="rounded-xl border border-border bg-card p-6">
        {activeTab === "summary" && summary && <SummaryContent summary={summary} />}
        {activeTab === "summary" && !summary && <NoDataMessage />}
        {activeTab === "highlights" && highlights && <HighlightsContent highlights={highlights} />}
        {activeTab === "highlights" && !highlights && <NoDataMessage />}
        {activeTab === "methodology" && methodology && <MethodologyContent methodology={methodology} />}
        {activeTab === "methodology" && !methodology && <NoDataMessage />}
      </div>
    </div>
  );
}

function NoDataMessage() {
  return (
    <p className="text-sm text-muted-foreground">
      Not available for this paper.
    </p>
  );
}

function SummaryContent({ summary }: { summary: PaperSummary }) {
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
              <p className="text-base leading-relaxed">{section.value}</p>
            </div>
          ),
      )}
    </div>
  );
}

function HighlightsContent({ highlights }: { highlights: PaperHighlights }) {
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
                  <li key={i} className="flex gap-2 text-base leading-relaxed">
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

function MethodologyContent({ methodology }: { methodology: MethodologyAnalysis }) {
  return (
    <div className="flex flex-col gap-4">
      {methodology.approach && (
        <div>
          <h3 className="mb-1 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Approach
          </h3>
          <p className="text-base leading-relaxed">{methodology.approach}</p>
        </div>
      )}

      {methodology.datasets.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Datasets
          </h3>
          <ul className="flex flex-col gap-1">
            {methodology.datasets.map((ds, i) => (
              <li key={i} className="flex gap-2 text-base">
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
          <p className="text-base leading-relaxed">{methodology.results}</p>
        </div>
      )}

      {methodology.statistical_significance && (
        <div>
          <h3 className="mb-1 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Statistical Significance
          </h3>
          <p className="text-base leading-relaxed">{methodology.statistical_significance}</p>
        </div>
      )}
    </div>
  );
}
