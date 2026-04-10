# quick-auditory-learning コスト計算見直し

## 目的
- コスト表と合計が、画面に見えている項目と一致するようにする。

## 変更
- `prefetch` は内部計測として保持するが、画面の合計や一覧の合計には含めない。
- `get_session_generation_costs()` と `list_playback_sessions()` の合計を、画面に出す cost kind だけで計算するようにした。
- `get_paper_generation_costs()` も表示対象 kind のみに絞った。
- 仕様にコスト方針を追記した。

## 確認
- `cd /workspace/quick-auditory-learning/backend && uv run python -m pytest ../../tests/test_quick_auditory_learning.py -k 'visible_generation_cost_totals_ignore_prefetch or get_session_generation_costs_ignores_prefetch_in_totals or search_by_vector_fetches_rows_before_cursor_closes'`
- `cd /workspace/quick-auditory-learning/frontend && npm test`
- `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`

