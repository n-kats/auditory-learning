# Implementation Status

## 優先度 A
- [ ] PDF 解析キューの並列度制御と監視ログ整備。
- [ ] Frontend `src/components/PlayerPanel.tsx` への自動スクロール実装。

## 優先度 B
- [x] `frontend/node_modules` の再利用と npm 起動高速化 (担当: Codex, 2026-04-02, 次: なし)
- [x] `uv sync` と frontend build の再実行抑制 (担当: Codex, 2026-04-02, 次: なし)
- [ ] `quick-auditory-learning/` の frontend へ audio playback を接続する。
- [ ] `quick-auditory-learning/` の compose 起動確認と backend API 接続。
- [ ] `quick-auditory-learning/` の起動時 JSONL 自動同期の実運用確認。
- [ ] `auditory_learning/utils/voice_utils.py` の VOICEVOX 接続再試行ロジック。
- [ ] `docs/images/` の最新 UI キャプチャ更新。

## 優先度 C
- [ ] pytest 向けモックデータの `_tmp/fixtures/` 整備。
- [ ] README へ日本語/英語のセットアップ手順追記。
- [x] `AGENTS.md` の `scripts/serve.sh` 説明削除 (担当: Codex, 2025-10-26, 次: なし)
- [x] `AGENTS.md` ワークフロー節の詳細追記 (担当: Codex, 2025-10-26, 次: なし)
- [x] `AGENTS.md` を `_local/templates/template_AGENTS.md` に寄せる (担当: Codex, 2026-04-03, 次: なし)
- [x] `AGENTS.md` の `## コミット & Pull Request` 節を除去する (担当: Codex, 2026-04-04, 次: なし)

> 状態更新時は担当者名、日付、次アクションを併記すること: `- [ ] タスク (担当: name, 2025-02-18, 次: foo)`。
