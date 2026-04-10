# quick-auditory-learning session select paper_id 修正

## TODO
- [x] `SessionClientMessage` に `paper_id` を追加する。
- [x] backend の `/health` はログ上 200 を確認する。
- [ ] 変更後の session select を再確認する。
- [ ] 実行中 backend を再起動して修正を反映する。

## 決定事項
- `select` メッセージは `paper_id` を持つので、backend 側の受信モデルに明示的に追加する。

## 未決
- なし。

## 確認手順
- `select` 時に `SessionClientMessage` の属性エラーが出ないことを確認する。
- 実行中プロセスは古いコードのままの可能性があるため、再起動後に再確認する。
