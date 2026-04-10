# 作業ログ

## 目的
- slash を含む paper_id の route 404 をテストで防ぐ

## 対応
- backend で memo / websocket / audio / explanation / favorite の route を検証する
- frontend で memo / explanation / favorite の URL エンコードを検証する

## 確認
- `cd quick-auditory-learning/frontend && npm test`
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`
- `python -m py_compile tests/test_quick_auditory_learning.py quick-auditory-learning/backend/src/quick_auditory_learning/main.py`

## 完了
- `cond-mat/0104435` のような paper_id のルート回帰をテストで防ぐようにした
