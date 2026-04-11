# quick-auditory-learning session room isolation test

## 変更
- backend の session room pending が session_id ごとに分離されることを固定する回帰テストを追加

## 確認
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`
- `cd /workspace/quick-auditory-learning/frontend && npm test`
- `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`

## ねらい
- 別セッションの `paper_search_updated` が現在のセッションに混ざらないことをテストで再発防止する
