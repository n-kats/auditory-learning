# 2026-04-10 16:28:07 quick-auditory-learning コスト集計見直し

## 目的
- 合計時間を単純加算ではなく、重複区間を除いた壁時計時間にする
- 先読み中の生成コストも合計へ反映する

## 対応
- `db.py` に生成コストの区間 union ヘルパーを追加
- `get_session_generation_costs()` と `list_playback_sessions()` の合計時間を壁時計時間へ変更
- `paper_costs` の合計時間と合計費用も記録済み生成データから再計算するよう変更
- 仕様書に合計時間の定義を追記

## 確認
- `tests/test_quick_auditory_learning.py` に壁時計時間の union テストを追加
- `get_session_generation_costs()` が prefetch を含めた費用を返すことを確認

