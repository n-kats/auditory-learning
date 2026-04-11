# quick-auditory-learning frontend session view state refactor

## 変更
- `paper_ready` / `paper_search_updated` / `session_started` / `session_next_candidate_updated` / `session_stopped` の表示状態遷移を `sessionViewState.ts` に切り出し
- 別セッション・別論文の検索更新を無視する条件を helper 化
- helper の unit test を追加
- `paper_ready` と `paper_search_updated` の到着順が前後しても検索結果が残ることを固定

## 確認
- `cd /workspace/quick-auditory-learning/frontend && npm test`
- `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
- `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`

## ねらい
- 検索結果の混線や持ち越しを React 画面全体ではなく純粋関数で再現できるようにする
- 検索結果の送信タイミングに左右されず、同じ paper の結果が最終的に表示されることをテストで確認する
