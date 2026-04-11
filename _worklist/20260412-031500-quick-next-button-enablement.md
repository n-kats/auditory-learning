## 目的
- `次へ進む` ボタンが押せない問題を修正する。
- セッションがあるときはボタンを押せるようにし、クリック時は再生停止せずに `next` を送る。
- `set_next_candidate` / `session_next_candidate_updated` / `next_candidate_paper_ids` に通信プロトコルを揃える。

## 変更
- `frontend/src/App.tsx`
  - `次へ進む` の画面側の追加変更を取り下げ、元の `canSendSessionAction` 判定に戻した。
- `frontend/src/sessionActionAvailability.ts`
  - `next` 専用の有効条件を削除。
  - `next` が選択済み候補のときのハイライト判定を追加。
- `frontend/src/sessionActionAvailability.test.ts`
  - 追加した `next` 判定テストを削除。
  - next 候補ハイライト判定のテストを追加。
- `frontend/src/SearchResultList.tsx`
  - 検索結果の「次に再生」ボタンの色変更を戻した。
- `docs/spec/quick_auditory_learning.md`
  - `set_next_candidate` は次に再生する候補の指定であることを明記した。
- `docs/spec/quick_auditory_learning_flow.md`
  - 次候補の表示は選択状態が分かること、`session_next_candidate_updated` で反映することを要件として追記した。
- `docs/spec/quick_auditory_learning_messages.md`
  - `set_next_candidate` / `session_next_candidate_updated` / `next_candidate_paper_ids` に合わせた。

## 確認
- `cd quick-auditory-learning/frontend && npm test`
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`
