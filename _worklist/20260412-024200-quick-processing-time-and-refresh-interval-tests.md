# quick-processing-time-and-refresh-interval-tests

- 目的
  - 全体処理時間が idle を含んで伸びないことを確認する。
  - `次へ進む` / `次に再生` の有効条件が loading に引っ張られないことを確認する。
  - セッション一覧の更新 polling が攻撃的になりすぎないことを確認する。

- 実施内容
  - backend に `generation_cost_wall_elapsed_ms_from_rows` の idle gap を 0 にしない回帰テストを追加。
  - frontend に `sessionActionAvailability` を追加し、`next` / `search result interact` の有効条件を pure helper で固定。
  - frontend に `backendDirectoryPolling` を追加し、health/session refresh の polling 間隔を緩和して固定。

- 確認
  - `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
  - `cd /workspace/quick-auditory-learning/frontend && npm test`
  - `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`

- 結果
  - すべて通過。
