# quick-auditory-learning audio ended/pause 分離

## 目的
- チャンク末尾で `pause` が発火して `isPlaying` が false になり、次チャンクの自動再生が止まる可能性を減らす。

## 変更
- `<audio>` の `onPause` で `ended` 時は `isPlaying` を落とさないように変更。

## 確認
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`

