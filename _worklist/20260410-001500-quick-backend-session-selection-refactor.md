# 作業ログ

## 目的
- quick-auditory-learning backend の候補選択まわりを `main.py` から切り出して薄くする

## 対応
- 次候補選択、検索モード整列、イベントからの復元を独立モジュールへ移す
- `main.py` は WebSocket と永続化の進行役に寄せる

## 確認
- backend の pytest と py_compile を通した
