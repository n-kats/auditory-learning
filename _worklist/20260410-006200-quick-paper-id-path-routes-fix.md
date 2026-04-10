# 作業ログ

## 目的
- slash を含む paper_id で memo/audio/favorite/explanation が 404 にならないようにする

## 対応
- backend の paper_id ルートを `path` converter に変更する
- frontend の paper_id を含む API 送信を `encodeURIComponent` する

## 確認
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`
- `python -m py_compile quick-auditory-learning/backend/src/quick_auditory_learning/main.py`

## 完了
- `cond-mat/0104435` のような paper_id を扱えるようにした
