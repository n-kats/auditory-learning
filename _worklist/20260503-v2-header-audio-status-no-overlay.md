# v2 header audio status / overlay cleanup

## 目的
- 画面中央に重なる状態メッセージを消す
- 音声状態をヘッダーの `db: ok` の横に移す
- タイトルを `AUDITORY LEARNING V2` に揃える

## 対応
- `SessionTopPanel` の状態チップ表示を削除
- `useDocumentSession` のページ読み込み文言を非表示化
- `App` のヘッダーに `音声:OK / 確認中 / 失敗` を表示
- `quick-auditory-learning` の表記が残っていないか確認

## 確認
- frontend build

## 結果
- `ページを表示しています` 系の重なり表示を削除
- 音声状態をヘッダーに集約
- `続きから` 一覧の current セッションを `再生中` で表示
- `開始` ボタンを quick の `start-button` 位置と見た目に合わせる
- 音声状態の箱を外してヘッダーのテキストに戻す
