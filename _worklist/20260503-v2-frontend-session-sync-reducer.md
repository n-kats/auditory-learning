# v2 frontend session sync reducer

## 目的
- `useDocumentSession` 内の WebSocket 受信処理を pure reducer に分離する
- quick っぽく、状態遷移をテスト可能にする

## 実施内容
- `src/documentSessionSync.ts` を追加
- `session_snapshot` / `page_updated` / `favorite_toggled` の受信状態遷移を pure 関数化
- `src/documentSessionSync.test.ts` を追加
- `useDocumentSession.ts` の ws 分岐を reducer 経由に変更

## 確認
- `cd /workspace/v2/frontend && npm test -- --run src/documentSessionSync.test.ts src/pageState.test.ts src/audioPreferences.test.ts`
- `cd /workspace/v2/frontend && npm run build`

## 状態
- 完了
