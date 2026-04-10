# quick-auditory-learning 先読みコスト記録修正

## 目的
- 先読みで生成した解説・音声の費用が 0 のままにならないようにする。

## 変更
- `_schedule_next_paper_prefetch()` で `generate_explanation()` に cost recorder を渡すようにした。
- 先読みで実際に発生した `explanation` / `audio` の費用を通常のコストとして記録する。
- 先読みで生成されたものが後から使われたときに、0 のまま表示される状態を減らす。

## 確認
- `cd /workspace/quick-auditory-learning/backend && uv run python -m pytest ../../tests/test_quick_auditory_learning.py -k 'schedule_next_paper_prefetch_records_generation_cost or get_session_generation_costs_ignores_prefetch_in_totals or visible_generation_cost_totals_ignore_prefetch'`

