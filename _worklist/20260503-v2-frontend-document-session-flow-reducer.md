# v2 frontend document session flow reducer

## 目的
- `useDocumentSession` の状態遷移を pure reducer でシミュレートできるようにする
- start / resume / page load / favorite / ws を state-only でテストする

## 実施内容
- `src/documentSessionState.ts` を追加
- `simulateDocumentSessionFlow` を追加
- `src/documentSessionState.test.ts` を追加

## 確認
- `cd /workspace/v2/frontend && npm test -- --run src/documentSessionState.test.ts src/documentSessionSync.test.ts src/pageState.test.ts src/audioPreferences.test.ts`
- `cd /workspace/v2/frontend && npm run build`

## 状態
- 完了
