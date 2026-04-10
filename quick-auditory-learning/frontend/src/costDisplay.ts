import type { SessionCostItem, SessionCosts } from "./api";

export type CostRow = {
  kind: string;
  elapsedLabel: string;
  elapsedWithoutPrefetchLabel: string;
  costLabel: string;
  isPending: boolean;
};

export const COST_ROW_ORDER = [
  "search",
  "embedding",
  "explanation",
  "audio",
  "keyword_generation",
  "query_generation",
] as const;

export function formatDurationSeconds(durationMs: number | null | undefined): string {
  if (durationMs === null || durationMs === undefined) {
    return "計算中";
  }
  return `${(durationMs / 1000).toFixed(1)}秒`;
}

export function formatUsd(costUsd: number | null | undefined): string {
  if (costUsd === null || costUsd === undefined) {
    return "計算中";
  }
  return `$${costUsd.toFixed(6)}`;
}

export function formatCostKind(kind: string): string {
  switch (kind) {
    case "search":
      return "検索";
    case "embedding":
      return "ベクトル化";
    case "explanation":
      return "解説";
    case "audio":
      return "音声生成";
    case "keyword_generation":
      return "キーワード列";
    case "query_generation":
      return "全文検索クエリ";
    default:
      return kind;
  }
}

export function buildCostRows(items: SessionCostItem[]): CostRow[] {
  const rows = new Map(items.map((item) => [item.kind, item]));
  return COST_ROW_ORDER.map((kind) => {
    const item = rows.get(kind);
    const isPending = item?.status === "pending" || item === undefined;
    return {
      kind: formatCostKind(kind),
      elapsedLabel: isPending ? "計算中" : formatDurationSeconds(item?.elapsed_ms ?? 0),
      elapsedWithoutPrefetchLabel: isPending
        ? "計算中"
        : formatDurationSeconds(item?.elapsed_ms_without_prefetch ?? 0),
      costLabel: isPending ? "計算中" : formatUsd(item?.estimated_cost_usd ?? 0),
      isPending,
    };
  });
}

export function formatAudioDurationNote(costs: SessionCosts | null, fallbackDurationMs: number | null): string | null {
  if (!costs) {
    return null;
  }
  const durationMs = costs.audio_duration_ms && costs.audio_duration_ms > 0 ? costs.audio_duration_ms : fallbackDurationMs;
  return `音声の長さ ${durationMs && durationMs > 0 ? formatDurationSeconds(durationMs) : "不明"}`;
}
