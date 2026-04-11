# quick-reduce-polling

- 目的
  - リクエスト数が多すぎる問題を減らす。

- 実施内容
  - health polling を 60 秒に変更。
  - session summary polling を 120 秒に変更。
  - polling 間隔を定数化し、テストで固定。

- 確認
  - `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
  - `cd /workspace/quick-auditory-learning/frontend && npm test`
  - `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`

- 結果
  - すべて通過。
