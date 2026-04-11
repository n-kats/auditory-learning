# quick-auditory-learning backend polling restore

## やること
- health polling と session refresh polling を元の間隔へ戻す

## 変更
- `HEALTH_POLL_INTERVAL_MS` を `10000` に戻す
- `SESSION_REFRESH_INTERVAL_MS` を `30000` に戻す
- 間隔値のテストを更新する

## 確認
- frontend の `vitest` / `tsc`

