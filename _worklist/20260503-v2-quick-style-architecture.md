# 20260503 v2 quick style architecture

## 目的
- v2 のコード構成と状態管理を quick 風に寄せる。
- `App.tsx` を肥大化させず、pure function と hook を中心に分割する。
- セッション同期や再開の方針を文書化して、会話が終わっても参照できるようにする。

## 決定事項
- Postgres を正本にする。
- WebSocket は複数端末同期と状態通知に使う。
- PDF / 画像 / 音声はファイルのままにする。
- メモは今回の対象外。
- `App.tsx` は orchestration に限定する。

## 作業項目
- [x] quick の実装構造を確認する。
- [x] v2 の実装方針を文書化する。
- [x] `docs/directory_structure.md` に v2 方針文書を追加する。
- [x] 作業ログを残す。

## 参照文書
- `docs/spec/v2_auditory_learning_architecture.md`
- `docs/directory_structure.md`

