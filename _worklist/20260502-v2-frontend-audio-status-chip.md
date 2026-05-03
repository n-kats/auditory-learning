# 20260502 v2 frontend audio status chip

## 目的
- `/explain/` と `/regenerate/` から返る音声状態をフロントで見えるようにする。

## 作業項目
- [x] 音声状態表示用の state を追加する。
- [x] `ExplainResponse` の `audio_status` / `audio_error` を表示に反映する。
- [x] デスクトップとモバイルの両方で状態が見えるようにする。
- [x] フロントのビルドで確認する。

## 確認結果
- `cd /workspace/v2/frontend && npm run build` 成功

## 確認手順
- [ ] `cd /workspace/v2/frontend && npm run build`
