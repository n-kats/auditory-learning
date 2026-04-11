# quick-processing-time-and-next-button

- 目的
  - 全体処理時間が再生待ち中にも増えて見える問題を直す。
  - `次へ` / `次に再生` が過度に `loading` で無効化される問題を緩和する。

- 実施内容
  - セッション一覧の `全体処理時間` を `total_generation_elapsed_ms` ベースで表示するよう変更。
  - プレイヤーの `次へ進む` ボタンから `loading` 依存を外した。
  - 検索結果一覧の `次に再生` / `次に再生候補` の `canInteract` 判定から `loading` を外した。

- 確認
  - `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
  - `cd /workspace/quick-auditory-learning/frontend && npm test`
  - `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`

- 結果
  - すべて通過。
