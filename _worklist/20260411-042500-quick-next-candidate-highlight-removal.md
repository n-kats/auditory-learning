# 2026-04-11 next candidate highlight removal
 
## 目的
- `次へ進む` ボタンや検索結果の next candidate 強調表示を削除する。
 
## TODO
- frontend のハイライト関連 state / helper / CSS を削除する。
- 関連テストを更新する。
- docs から強調表示の文言を削除する。
 
## 確認
- frontend の test / tsc を実行する。

## 完了
- `次へ進む` ボタンの `is-active` 強調を削除した。
- 検索結果の next candidate 強調表示を削除した。
- `shouldHighlightNextCandidateAction` と関連テストを削除した。
- docs から next candidate の緑系強調表示の文言を削除した。
- frontend の `npm test` と `npx tsc --noEmit` を確認する。
