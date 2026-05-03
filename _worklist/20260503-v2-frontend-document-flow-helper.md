# v2 frontend document flow helper

## 目的
- `useDocumentSession` から document 読み込み・再読み込みの手順を切り出す
- quick 風に、通信と状態更新の境界を明確にする

## 実施内容
- `src/documentSessionFlow.ts` を追加
- `retryInitDocumentWithBackoff` を追加
- `loadDocumentPage` を追加
- `useDocumentSession.ts` から page load / init retry の中身を分離

## 確認
- `cd /workspace/v2/frontend && npm test -- --run src/documentSessionSync.test.ts src/pageState.test.ts src/audioPreferences.test.ts`
- `cd /workspace/v2/frontend && npm run build`

## 状態
- 完了
