import { describe, expect, it } from "vitest";
import { resolveDisplayedNextCandidatePaperId } from "./nextCandidateDisplay";

describe("nextCandidateDisplay", () => {
  it("prefers the optimistic local selection over the backend next paper id", () => {
    expect(
      resolveDisplayedNextCandidatePaperId({
        selectedNextCandidatePaperId: "paper-local",
        nextPaperId: "paper-backend",
      }),
    ).toBe("paper-local");
  });

  it("falls back to the backend next paper id when no local selection exists", () => {
    expect(
      resolveDisplayedNextCandidatePaperId({
        selectedNextCandidatePaperId: null,
        nextPaperId: "paper-backend",
      }),
    ).toBe("paper-backend");
  });
});
