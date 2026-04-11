## 対応
- `開始URLを入力して session を開始してください。` が session 確立後に出ないよう、表示分岐を修正する。
- `currentSessionId` があるのに start フォールバックへ落ちないようにする。

## 確認
- frontend の unit test を追加する。
- `npm test` と `npx tsc --noEmit` を通す。
