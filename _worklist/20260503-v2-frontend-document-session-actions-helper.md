# v2 frontend document session actions helper

## 目的
- `useDocumentSession` から開始・再開・ページ移動・再生成・お気に入り切替を切り出す
- quick 風に、hook を state 配線中心に寄せる

## 実施内容
- `src/documentSessionActions.ts` を追加
- start / resume / move / jump / regenerate / favorite を helper 化
- `useDocumentSession.ts` の下半分を helper 呼び出しに変更

## 確認
- `cd /workspace/v2/frontend && npm test -- --run src/documentSessionSync.test.ts src/pageState.test.ts src/audioPreferences.test.ts`
- `cd /workspace/v2/frontend && npm run build`

## 状態
- 完了
