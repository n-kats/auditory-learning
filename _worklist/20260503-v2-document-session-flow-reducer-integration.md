# v2 document session flow reducer integration

## 目的
- `useDocumentSession` の状態更新を reducer 経由へ寄せる
- WebSocket 受信とページ読み込みの状態遷移を state-only で追えるようにする

## 作業内容
- `documentSessionState.ts` の flow reducer を整理
- `useDocumentSession.ts` で `flowState` を持ち、`dispatchFlowEvent` から更新する構成へ変更
- `documentSessionActions.ts` から状態の直接更新を外し、イベント駆動に寄せる

## 確認
- `cd /workspace/v2/frontend && npm test -- --run src/documentSessionState.test.ts src/documentSessionSync.test.ts src/pageState.test.ts src/audioPreferences.test.ts`
- `cd /workspace/v2/frontend && npm run build`

## 状態
- 完了
