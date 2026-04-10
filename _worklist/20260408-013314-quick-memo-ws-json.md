# quick-auditory-learning メモ WS JSON 化

## TODO
- [x] `memo/ws` の初期スナップショットを JSON 互換にする。
- [x] `datetime` が WS 送信で落ちないことを確認する。

## 決定事項
- `memo/ws` で送る payload は `send_json` にそのまま渡せる形に統一する。

## 未決
- なし。

## 確認手順
- メモ WS 接続時に `TypeError: Object of type datetime is not JSON serializable` が出ないことを確認する。
- `python -m py_compile quick-auditory-learning/backend/src/quick_auditory_learning/main.py` が通ることを確認する。
