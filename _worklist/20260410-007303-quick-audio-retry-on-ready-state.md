# quick-auditory-learning audio 再試行

## 目的
- 続きから直後やチャンク切り替え直後に、音声要素がまだ再生可能になっていない場合でも再生を始められるようにする。

## 変更
- `useAudioPlayback` で `play()` 失敗時の再試行を、`canplay` / `loadeddata` / `loadedmetadata` 到達時にも行うようにした。

## 確認
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`

