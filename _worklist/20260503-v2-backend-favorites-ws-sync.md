# v2 backend favorites / ws sync

## 目的
- document を favorite できるようにする
- session の状態変化を WebSocket で同期できるようにする
- quick と同じく、状態遷移を pure function でテストする

## 対応方針
- favorite は Postgres の `favorites` テーブルで保持する
- ws は session イベントの送受信口として追加する
- 状態遷移ロジックは pure function に切り出してテストする
- simulator ベースのテストを追加して、イベント列での遷移を検証する

## 確認
- backend test

## 結果
- `favorites` テーブルと toggle/list API を追加
- `sessions/ws` の push 配信を追加
- pure reducer `session_sync.py` と simulator テストを追加
- repository と reducer のテストを通過
