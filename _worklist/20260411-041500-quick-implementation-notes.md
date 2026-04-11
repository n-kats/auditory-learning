# 20260411 quick-auditory-learning implementation notes

## 目的
quick-auditory-learning の実装で踏みやすい注意点を、仕様とは別の文書として整理する。

## 反映内容
- `docs/spec/quick_auditory_learning_implementation_notes.md` を新規作成した。
- `docs/spec/quick_auditory_learning.md` の関連文書に追記した。
- `docs/spec/仕様.md` の仕様一覧に追記した。
- `docs/directory_structure.md` に置き場を追記した。

## 注意として残した項目
- `session_started` と `paper_ready` の役割分担。
- `paper_ready` と `paper_search_updated` の順序差分。
- `paper_search_updated` の stale 判定。
- `next_paper_id` は単一値で、queue ではないこと。
- 先読み開始タイミングと stale prefetch の破棄。
- 仕様にない fallback を入れないこと。

## 確認
- ドキュメント更新のみ。
