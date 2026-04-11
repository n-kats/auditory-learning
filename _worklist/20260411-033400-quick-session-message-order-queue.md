## 作業内容
- quick-auditory-learning frontend の WebSocket メッセージを直列キューで処理するように変更
- `session_started` / `paper_ready` / `session_costs_updated` などが受信順に処理されることをテスト追加
- session 切り替え時の古い非同期処理の影響を受けにくくするため、`onError` を ref 経由に変更

## 確認
- `cd /workspace/quick-auditory-learning/frontend && npm test`
- `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
- `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`

## 結果
- frontend tests: 17 passed
- TypeScript check: passed
- backend tests: 54 passed

## 追記
- next_paper_id が決まったら、次論文の検索も先読みする修正を追加
- 検索結果を paper_ready で先に使えるようにした
