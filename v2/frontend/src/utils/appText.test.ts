import { describe, expect, it } from "vitest";

import { buildPaperLabel, buildUploadedPdfFileName } from "./appText";

describe("buildPaperLabel", () => {
  it("formats uploaded pdf source urls", () => {
    expect(buildPaperLabel("upload://session-1/sample.pdf")).toBe("uploaded sample.pdf");
  });

  it("formats arxiv urls as before", () => {
    expect(buildPaperLabel("https://arxiv.org/pdf/2604.16347")).toBe("arXiv 2604.16347");
  });

  it("extracts uploaded pdf file names", () => {
    expect(buildUploadedPdfFileName("upload://session-1/sample.pdf")).toBe("sample.pdf");
  });
});
