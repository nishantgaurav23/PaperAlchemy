"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, FileText, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { MAX_FILE_SIZE, ACCEPTED_FILE_TYPE } from "@/types/upload";

interface DropZoneProps {
  onFileSelect: (file: File) => void;
  selectedFile?: File | null;
  disabled?: boolean;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function DropZone({ onFileSelect, selectedFile, disabled }: DropZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): string | null => {
    if (file.type !== ACCEPTED_FILE_TYPE) {
      return "Only PDF files are accepted. Please select a .pdf file.";
    }
    if (file.size > MAX_FILE_SIZE) {
      return "File exceeds 50MB limit. Please select a smaller file.";
    }
    return null;
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }
      setError(null);
      onFileSelect(file);
    },
    [onFileSelect, validateFile],
  );

  const handleDragEnter = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!disabled) setIsDragOver(true);
    },
    [disabled],
  );

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!disabled) setIsDragOver(true);
    },
    [disabled],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      if (disabled) return;

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        // Find first PDF, or use first file for validation error
        const pdfFile = Array.from(files).find((f) => f.type === ACCEPTED_FILE_TYPE);
        handleFile(pdfFile ?? files[0]);
      }
    },
    [disabled, handleFile],
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        handleFile(files[0]);
      }
      // Reset input so same file can be re-selected
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [handleFile],
  );

  const handleBrowseClick = useCallback(() => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  }, [disabled]);

  return (
    <div className="flex flex-col gap-3">
      <div
        data-testid="drop-zone"
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleBrowseClick}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed p-12 transition-all",
          "border-border bg-muted/30 hover:border-muted-foreground/50 hover:bg-muted/50",
          isDragOver && "border-primary bg-primary/5",
          disabled && "pointer-events-none opacity-50",
          error && "border-destructive/50",
        )}
      >
        {selectedFile ? (
          <div className="flex items-center gap-3">
            <FileText className="size-8 text-primary" />
            <div>
              <p className="font-medium">{selectedFile.name}</p>
              <p className="text-sm text-muted-foreground">{formatFileSize(selectedFile.size)}</p>
            </div>
          </div>
        ) : (
          <>
            <div
              data-testid="upload-icon"
              className={cn(
                "rounded-full p-4 transition-colors",
                isDragOver ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground",
              )}
            >
              <Upload className="size-8" />
            </div>
            <div className="text-center">
              <p className="text-lg font-medium">
                Drag and drop your PDF here
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                or{" "}
                <span className="font-medium text-primary underline-offset-4 hover:underline">
                  browse
                </span>{" "}
                to select a file
              </p>
              <p className="mt-2 text-xs text-muted-foreground">
                PDF files only, up to 50MB
              </p>
            </div>
          </>
        )}
      </div>

      <input
        ref={fileInputRef}
        data-testid="file-input"
        type="file"
        accept=".pdf,application/pdf"
        onChange={handleInputChange}
        className="hidden"
        disabled={disabled}
      />

      {error && (
        <div className="flex items-center gap-2 rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">
          <X className="size-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
