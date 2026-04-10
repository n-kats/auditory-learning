import { describe, expect, it } from "vitest";

import { buildCostRows, formatAudioDurationNote, formatCostKind, formatDurationSeconds, formatUsd } from "./costDisplay";

describe("costDisplay helpers", () => {
  it("formats duration and cost values", () => {
    expect(formatDurationSeconds(1500)).toBe("1.5秒");
    expect(formatUsd(0.1234567)).toBe("$0.123457");
  });

  it("formats cost kind labels", () => {
    expect(formatCostKind("audio")).toBe("音声生成");
    expect(formatCostKind("unknown")).toBe("unknown");
  });

  it("builds cost rows in a fixed order", () => {
    const rows = buildCostRows([
      { kind: "audio", elapsed_ms: 500, elapsed_ms_without_prefetch: 400, estimated_cost_usd: 0.01 },
      { kind: "search", elapsed_ms: 100, elapsed_ms_without_prefetch: 80, estimated_cost_usd: 0.001 },
    ]);
    expect(rows[0]).toEqual({
      kind: "検索",
      elapsedLabel: "0.1秒",
      elapsedWithoutPrefetchLabel: "0.1秒",
      costLabel: "$0.001000",
      isPending: false,
    });
    expect(rows[3]).toEqual({
      kind: "音声生成",
      elapsedLabel: "0.5秒",
      elapsedWithoutPrefetchLabel: "0.4秒",
      costLabel: "$0.010000",
      isPending: false,
    });
  });

  it("marks missing cost rows as pending", () => {
    const rows = buildCostRows([]);
    expect(rows[0]).toEqual({
      kind: "検索",
      elapsedLabel: "計算中",
      elapsedWithoutPrefetchLabel: "計算中",
      costLabel: "計算中",
      isPending: true,
    });
  });

  it("formats the audio duration note", () => {
    expect(formatAudioDurationNote({ audio_duration_ms: 900, total_elapsed_ms: 0, total_wall_elapsed_ms: 0, total_cost_usd: 0, items: [], session_id: "s" }, null)).toBe(
      "音声の長さ 0.9秒",
    );
    expect(formatAudioDurationNote({ audio_duration_ms: null, total_elapsed_ms: 0, total_wall_elapsed_ms: 0, total_cost_usd: 0, items: [], session_id: "s" }, 1200)).toBe(
      "音声の長さ 1.2秒",
    );
    expect(formatAudioDurationNote(null, 1200)).toBeNull();
  });
});
