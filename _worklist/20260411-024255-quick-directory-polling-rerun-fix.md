# quick-auditory-learning directory polling rerun fix

## やること
- `useBackendDirectoryData` の effect が `onError` / `onSuccess` の参照変更で再起動しないようにする
- その結果、`/health` や `/favorites` / `/history/recent` / `/sessions/recent` の連打を止める

## 対応
- callbacks を ref 経由で参照するように変更
- effect の依存配列から callback を外した

## 確認
- frontend の `tsc` / `vitest`
- backend の pytest

