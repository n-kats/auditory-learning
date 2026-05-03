# v2 frontend workspace layout hook

## 目的
- `useDocumentSession` から画面幅判定と分割ドラッグを切り出す
- quick 風に、レイアウト副作用を別 hook に分離する

## 実施内容
- `src/hooks/useWorkspaceLayout.ts` を追加
- `resize` によるモバイル判定を移動
- divider の pointer move / up / cancel を移動
- `useDocumentSession.ts` から workspace 由来の useEffect を削除

## 確認
- `cd /workspace/v2/frontend && npm test -- --run src/documentSessionSync.test.ts src/pageState.test.ts src/audioPreferences.test.ts`
- `cd /workspace/v2/frontend && npm run build`

## 状態
- 完了
