import { describe, it, expect, vi, beforeEach } from "vitest";
import { copyToClipboard, downloadFile } from "./clipboard";

describe("copyToClipboard", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("copies text using navigator.clipboard.writeText", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    const result = await copyToClipboard("hello world");
    expect(writeText).toHaveBeenCalledWith("hello world");
    expect(result).toBe(true);
  });

  it("returns false when clipboard API fails", async () => {
    const writeText = vi.fn().mockRejectedValue(new Error("denied"));
    Object.assign(navigator, { clipboard: { writeText } });

    const result = await copyToClipboard("hello world");
    expect(result).toBe(false);
  });
});

describe("downloadFile", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("creates a blob URL and triggers download", () => {
    const mockUrl = "blob:http://localhost/fake";
    const createObjectURL = vi.fn().mockReturnValue(mockUrl);
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });

    const clickSpy = vi.fn();
    const mockAnchor = { href: "", download: "", click: clickSpy } as unknown as HTMLAnchorElement;
    vi.spyOn(document, "createElement").mockReturnValue(mockAnchor);

    downloadFile("content here", "export.bib");

    expect(createObjectURL).toHaveBeenCalled();
    expect(mockAnchor.download).toBe("export.bib");
    expect(clickSpy).toHaveBeenCalled();
    expect(revokeObjectURL).toHaveBeenCalledWith(mockUrl);
  });
});
