import { describe, expect, it } from "vitest";

import { normalizeMemoText, shouldSaveMemo } from "./paperMemo";

describe("paperMemo helpers", () => {
  it("normalizes nullish memo text to an empty string", () => {
    expect(normalizeMemoText(null)).toBe("");
    expect(normalizeMemoText(undefined)).toBe("");
    expect(normalizeMemoText("memo")).toBe("memo");
  });

  it("requires a dirty memo and a paper id before saving", () => {
    expect(
      shouldSaveMemo({
        currentPaperId: null,
        isDirty: true,
        memo: "memo",
        remoteValue: "",
      }),
    ).toBe(false);
    expect(
      shouldSaveMemo({
        currentPaperId: "p-1",
        isDirty: false,
        memo: "memo",
        remoteValue: "",
      }),
    ).toBe(false);
    expect(
      shouldSaveMemo({
        currentPaperId: "p-1",
        isDirty: true,
        memo: "memo",
        remoteValue: "memo",
      }),
    ).toBe(false);
    expect(
      shouldSaveMemo({
        currentPaperId: "p-1",
        isDirty: true,
        memo: "changed",
        remoteValue: "memo",
      }),
    ).toBe(true);
  });
});
