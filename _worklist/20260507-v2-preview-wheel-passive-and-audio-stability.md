- 目的:
  - v2 で表示される `Unable to preventDefault inside passive event listener invocation` を解消する。
  - 音声再生が不安定になる要因を減らす。

- 方針:
  - preview の wheel 処理は React の synthetic event から外し、`passive: false` の native listener で扱う。
  - audio の `play()` は音量・再生速度変更とは切り離し、src と speaker 設定変化に限定する。

- 確認:
  - `npm test` は成功。
  - `npm run build` は成功。
  - `npm run lint` はこの frontend に script が無かったため実行不可。

- 完了:
  - `preview-stage` の wheel listener を `passive: false` の native listener に切り替えた。
  - audio の `play()` 再実行条件を整理して再生安定性を上げた。
