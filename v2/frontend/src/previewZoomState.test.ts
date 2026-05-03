import { describe, expect, it } from "vitest";

import { adjustPreviewZoomForWheel, clampPreviewZoom } from "./previewZoomState";

describe("previewZoomState", () => {
  it("clamps preview zoom to a safe range", () => {
    expect(clampPreviewZoom(Number.NaN)).toBe(1);
    expect(clampPreviewZoom(0.1)).toBe(0.5);
    expect(clampPreviewZoom(5)).toBe(3);
  });

  it("zooms in when the wheel moves up and out when it moves down", () => {
    expect(adjustPreviewZoomForWheel(1, -1)).toBe(1.15);
    expect(adjustPreviewZoomForWheel(1, 1)).toBe(0.85);
  });

  it("keeps the zoom within bounds", () => {
    expect(adjustPreviewZoomForWheel(3, -1)).toBe(3);
    expect(adjustPreviewZoomForWheel(0.5, 1)).toBe(0.5);
  });
});
