# quick-auditory-learning session autoplay off

## やること
- セッションに入った直後の自動再生を止める

## 変更
- `handleStart` は `shouldAutoPlay=false` にする
- `handleResumeSession` の復元後は再生中にしない
- 仕様に「セッションに入った直後は自動再生しない」を追記

## 確認
- frontend の `vitest` / `tsc`

