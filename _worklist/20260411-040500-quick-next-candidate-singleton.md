# 2026-04-11 next candidate 単数化

## 変更
- `next_candidate_paper_ids` を廃止し、`next_paper_id` に一本化した。
- backend の `session_next_candidate_updated` と `paper_ready` からリストを削除した。
- frontend の session state / replay state / tests から `nextCandidatePaperIds` を削除した。

## 確認
- frontend の `npm test`
- frontend の `npx tsc --noEmit`
- backend の pytest

