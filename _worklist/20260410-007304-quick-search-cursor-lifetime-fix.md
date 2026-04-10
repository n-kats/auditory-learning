# quick-auditory-learning search cursor lifetime 修正

## 目的
- `search_by_vector()` がカーソルを閉じたあとに `fetchall()` していた不具合を直す。

## 変更
- `quick_auditory_learning.search.search_by_vector()` の `fetchall()` を `with conn.cursor()` の内側に戻した。
- カーソルが閉じたあとに `fetchall()` すると失敗する fake cursor テストを追加した。

## 確認
- `cd /workspace/quick-auditory-learning/backend && uv run python -m pytest ../../tests/test_quick_auditory_learning.py -k 'search_by_vector_fetches_rows_before_cursor_closes or search_papers_falls_back_to_db_random_candidates'`
- `python -m py_compile /workspace/quick-auditory-learning/backend/src/quick_auditory_learning/search.py /workspace/tests/test_quick_auditory_learning.py`

