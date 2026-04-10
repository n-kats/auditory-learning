# quick-auditory-learning API 失敗通知表示

## 目的
- OpenAI / VoiceVox などの API が使えなかったとき、その事実を画面にも表示する。

## 変更
- `paper_ready` に `notices` を追加。
- backend 側で検索キーワード生成、全文検索クエリ生成、埋め込み生成、音声生成の失敗を notice として集約。
- frontend 側で notice カードを表示。
- session replay にも notice を保持。

## 確認
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`
- `cd /workspace/quick-auditory-learning/backend && uv run python -m pytest ../../tests/test_quick_auditory_learning.py -k 'paper_ready_payload_collects_api_failure_notices or paper_ready_payload_keeps_search_and_force_flag or search_by_vector_fetches_rows_before_cursor_closes'`

