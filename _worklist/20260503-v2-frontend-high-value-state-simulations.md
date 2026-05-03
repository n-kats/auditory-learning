# v2 frontend high value state simulations

## 目的
- 状態だけで追える高価値のシミュレーションを先に追加する
- 古い読み込み結果の無視、resume 復元、favorite 同期を固定する

## 実施内容
- stale load を識別する load_id を reducer に追加
- resume 復元のシミュレーションを追加
- favorite 同期のシミュレーションを追加

## 確認
- `cd /workspace/v2/frontend && npm test -- --run src/documentSessionState.test.ts src/documentSessionSync.test.ts src/pageState.test.ts src/audioPreferences.test.ts`
- `cd /workspace/v2/frontend && npm run build`

## 状態
- 完了
