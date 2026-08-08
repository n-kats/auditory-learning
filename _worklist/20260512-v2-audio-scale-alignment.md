## 2026-05-12 v2 audio scale alignment

### 目的
- v2 の再生スライダーを quick と同じレンジにそろえる。
- v2 の VOICEVOX 生成パラメータは 1.0 にする。

### 変更
- frontend の音量・速度スライダー範囲を quick に合わせた。
- frontend の保存・復元の上限も quick に合わせた。
- backend の VOICEVOX speaker 生成時に speed / volume を明示せず、既定の 1.0 を使うようにした。
- 停止後に再生を押したとき、続きから再開するようにした。

### 確認
- これから実行する。
