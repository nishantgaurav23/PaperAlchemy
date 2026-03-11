"use client";

import { useCallback, useState } from "react";
import { AlertCircle, RotateCcw } from "lucide-react";
import { DropZone } from "@/components/upload/drop-zone";
import { UploadProgress } from "@/components/upload/upload-progress";
import { AnalysisResults } from "@/components/upload/analysis-results";
import { uploadPdf } from "@/lib/api/upload";
import { Button } from "@/components/ui/button";
import type { UploadResponse, UploadStatus } from "@/types/upload";

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = useCallback(async (file: File) => {
    setSelectedFile(file);
    setError(null);
    setStatus("uploading");

    try {
      setStatus("processing");
      const response = await uploadPdf(file);
      setResult(response);
      setStatus("complete");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed. Please try again.");
      setStatus("error");
    }
  }, []);

  const handleUploadAnother = useCallback(() => {
    setSelectedFile(null);
    setStatus("idle");
    setResult(null);
    setError(null);
  }, []);

  const handleRetry = useCallback(() => {
    if (selectedFile) {
      setError(null);
      handleFileSelect(selectedFile);
    }
  }, [selectedFile, handleFileSelect]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold tracking-tight">Upload Paper</h1>
        <p className="text-sm text-muted-foreground">
          Upload a PDF to get AI-generated summary, highlights, and methodology analysis.
        </p>
      </div>

      {status === "idle" && (
        <DropZone onFileSelect={handleFileSelect} />
      )}

      {(status === "uploading" || status === "processing") && selectedFile && (
        <UploadProgress status={status} fileName={selectedFile.name} />
      )}

      {status === "error" && (
        <div className="flex flex-col items-center gap-4 rounded-xl border border-destructive/30 bg-destructive/5 p-8">
          <AlertCircle className="size-10 text-destructive" />
          <div className="text-center">
            <p className="font-medium text-destructive">Upload Failed</p>
            <p className="mt-1 text-sm text-muted-foreground">{error}</p>
          </div>
          <div className="flex gap-3">
            <Button variant="outline" onClick={handleUploadAnother}>
              Try Different File
            </Button>
            <Button onClick={handleRetry}>
              <RotateCcw className="size-4" />
              Retry
            </Button>
          </div>
        </div>
      )}

      {status === "complete" && result && (
        <AnalysisResults data={result} onUploadAnother={handleUploadAnother} />
      )}
    </div>
  );
}
