## 目的
- quick-auditory-learning の通信フロー文書に残っていた矛盾を解消する。

## 変更
- `docs/spec/quick_auditory_learning_flow.md`
  - 後半に残っていた「`next_paper_id` が決まった時点で検索先読みする」という記述を削除。
  - `session_playback_started` 受信後に検索先読みする方針へ統一。

## 確認
- 文書内の `search prefetch` の開始条件が一貫して `session_playback_started` 基準になっていることを確認。

