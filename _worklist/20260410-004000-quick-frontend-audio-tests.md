# 作業ログ

## 目的
- quick-auditory-learning frontend の音声まわりの純粋関数テストを追加する

## 対応
- `audioPlayback` の clamp / load helper を切り出す
- `resolveAudioSourceUrl` をテスト可能にする
- vitest を導入して frontend 側からテストを実行できるようにする

## 確認
- `cd quick-auditory-learning/frontend && npm test`
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`

## 完了
- frontend で最小の unit test 実行基盤を整えた
- 音声再生まわりの純粋関数テストを追加した
