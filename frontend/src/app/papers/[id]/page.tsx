"use client";

import { useCallback, useEffect, useState, useReducer } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, AlertCircle, FileQuestion } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PaperHeader } from "@/components/paper/paper-header";
import { PaperSections } from "@/components/paper/paper-sections";
import { PaperAnalysis } from "@/components/paper/paper-analysis";
import { RelatedPapers } from "@/components/paper/related-papers";
import { PaperDetailSkeleton } from "@/components/paper/paper-detail-skeleton";
import { getPaper, getRelatedPapers, requestAnalysis } from "@/lib/api/papers";
import type { PaperDetail } from "@/types/paper";
import type { Paper } from "@/types/paper";

type PageState =
  | { status: "loading" }
  | { status: "success"; paper: PaperDetail }
  | { status: "error"; message: string }
  | { status: "not-found" };

type Action =
  | { type: "loading" }
  | { type: "success"; paper: PaperDetail }
  | { type: "error"; message: string }
  | { type: "not-found" };

function reducer(_state: PageState, action: Action): PageState {
  switch (action.type) {
    case "loading":
      return { status: "loading" };
    case "success":
      return { status: "success", paper: action.paper };
    case "error":
      return { status: "error", message: action.message };
    case "not-found":
      return { status: "not-found" };
  }
}

export default function PaperDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [state, dispatch] = useReducer(reducer, { status: "loading" });
  const [relatedPapers, setRelatedPapers] = useState<Paper[]>([]);
  const [retryCount, setRetryCount] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    let cancelled = false;

    dispatch({ type: "loading" });

    getPaper(id).then(
      (data) => {
        if (!cancelled) dispatch({ type: "success", paper: data });
      },
      (err: unknown) => {
        if (cancelled) return;
        const status = (err as { status?: number }).status;
        if (status === 404) {
          dispatch({ type: "not-found" });
        } else {
          dispatch({
            type: "error",
            message: err instanceof Error ? err.message : "Failed to load paper",
          });
        }
      },
    );

    getRelatedPapers(id).then(
      (data) => {
        if (!cancelled) setRelatedPapers(data.papers);
      },
      () => {
        if (!cancelled) setRelatedPapers([]);
      },
    );

    return () => {
      cancelled = true;
    };
  }, [id, retryCount]);

  const handleRetry = () => setRetryCount((c) => c + 1);

  const handleRequestAnalysis = useCallback(async () => {
    setIsAnalyzing(true);
    try {
      const updatedPaper = await requestAnalysis(id);
      dispatch({ type: "success", paper: updatedPaper });
    } catch {
      // Analysis failed but paper is still visible
    } finally {
      setIsAnalyzing(false);
    }
  }, [id]);

  return (
    <div className="mx-auto max-w-5xl px-4 md:px-8 py-6">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => router.back()}
        className="mb-4"
        aria-label="Back"
      >
        <ArrowLeft className="size-4" />
        Back
      </Button>

      {state.status === "loading" && <PaperDetailSkeleton />}

      {state.status === "not-found" && (
        <div className="flex flex-col items-center gap-3 py-12 text-center">
          <FileQuestion className="size-12 text-muted-foreground" />
          <h2 className="text-lg font-semibold">Paper not found</h2>
          <p className="text-sm text-muted-foreground">
            The paper you are looking for does not exist or has been removed.
          </p>
          <Button variant="outline" onClick={() => router.back()}>
            Go Back
          </Button>
        </div>
      )}

      {state.status === "error" && (
        <div className="flex flex-col items-center gap-3 py-12 text-center">
          <AlertCircle className="size-12 text-destructive" />
          <h2 className="text-lg font-semibold">Failed to load paper</h2>
          <p className="text-sm text-muted-foreground">{state.message}</p>
          <Button variant="outline" onClick={handleRetry}>
            Retry
          </Button>
        </div>
      )}

      {state.status === "success" && (
        <div className="flex flex-col gap-8">
          <PaperHeader paper={state.paper} />

          <PaperSections sections={state.paper.sections} />

          <PaperAnalysis
            summary={state.paper.summary}
            highlights={state.paper.highlights}
            methodology={state.paper.methodology}
            onRequestAnalysis={handleRequestAnalysis}
            isAnalyzing={isAnalyzing}
          />

          <RelatedPapers papers={relatedPapers} />
        </div>
      )}
    </div>
  );
}
