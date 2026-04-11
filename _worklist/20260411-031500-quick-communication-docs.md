# quick-auditory-learning 通信仕様整備

## 目的
- session の種類、通信データ型、通信フローを漏れなく文書化する。
- `session_started` / `paper_ready` / `paper_search_updated` の順序差分を前提として整理する。

## 実施内容
- `docs/spec/quick_auditory_learning_messages.md` を新規作成する。
- `docs/spec/quick_auditory_learning_flow.md` を新規作成する。
- `docs/spec/quick_auditory_learning.md` に関連文書へのリンクを追加する。
- `docs/spec/仕様.md` に新規仕様書を追加する。
- `docs/directory_structure.md` に新規仕様書の置き場を追記する。
- `frontend/src/sessionPanelState.ts` を追加し、session 画面の表示モードを純粋関数化する。
- `frontend/src/sessionPanelState.test.ts` を追加し、loading / paper / start の分岐を固定する。

## 確認
- `session_started` だけでは start 文言に戻らず、`paper_ready` 到着で paper panel に遷移する前提をテストで固定する。
- 既存の session state / message state / replay のテストと合わせて、イベント順の回帰を防ぐ。
- `frontend` の `npx tsc --noEmit` を通した。
- `frontend` の `npm test` を通した。
