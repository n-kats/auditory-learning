# 20260410 quick-auditory-learning headset control sync

## 目的
ヘッドセットの play / pause 操作に UI の再生状態を同期させる。

## 変更
- `<audio>` の `onPlay` / `onPause` で `isPlaying` を更新した。
- `Media Session API` の `play` / `pause` / `stop` を登録した。
- `pause` は位置を戻さず停止、`stop` は先頭へ戻す形に分離した。

## 確認
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`

## 補足
- Media Session をサポートしないブラウザでは従来どおり audio 要素依存になる。
