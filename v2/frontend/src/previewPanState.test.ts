import { describe, expect, it } from "vitest";

import { applyPreviewDrag } from "./previewPanState";

describe("previewPanState", () => {
  it("moves the preview pan by the drag delta", () => {
    expect(applyPreviewDrag({ x: 10, y: -4 }, 3, 8)).toEqual({ x: 13, y: 4 });
  });

  it("returns the origin for invalid deltas", () => {
    expect(applyPreviewDrag({ x: 10, y: -4 }, Number.NaN, 8)).toEqual({ x: 10, y: -4 });
  });
});
