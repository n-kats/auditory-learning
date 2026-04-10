# 作業ログ

## 目的
- セッション開始直後だけ動きが不安定になる問題を切り分ける

## 対応
- 古い WebSocket の `onclose` / `onmessage` / `onopen` が、新しいセッションに割り込まないように、`socketRef.current` との同一性を確認して無視するようにした

## 確認
- frontend の型チェックを行う

