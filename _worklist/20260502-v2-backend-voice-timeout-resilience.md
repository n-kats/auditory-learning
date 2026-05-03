# 20260502 v2 backend voice timeout resilience

## 目的
- `/explain/` の解説生成が、VOICEVOX のタイムアウトで 500 にならないようにする。
- 音声生成失敗は解説返却とは分離し、後続の `/audio/` で再試行できる前提を維持する。

## 作業項目
- [x] `voice_utils.py` の VOICEVOX 呼び出しに明示的な timeout を入れる。
- [x] 音声生成失敗を `/explain/` の例外にしないようにする。
- [x] 必要ならログだけ出して解説テキストの返却を継続する。
- [x] `AUDITORY_LEARNING_V2_VOICEVOX_URL` と `AUDITORY_LEARNING_V2_FALLBACK_VOICEVOX_URL` を分ける。
- [x] 無効な URL の場合は fallback を使うようにする。
- [x] Docker Compose と README の変数名を合わせる。
- [x] 無効だった場合に採用 URL をログに出す。
- [x] バックエンドのビルド/テストで確認する。

## 確認手順
- [ ] `cd /workspace/v2/backend && python -m compileall src`
- [ ] 可能なら `pytest` で関連テストを実行する。
