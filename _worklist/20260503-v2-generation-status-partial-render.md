- 目的
  - 読み込み中をぐるぐるで表し、解説とプレビューは先に描画できるなら先に描画する。
  - 生成開始と完了を ws で受け取り、ヘッダーの `音声:ok` の横に小さく生成中表示を出す。

- 実装
  - backend の `generation_task` で、cache hit でない実生成時のみ `generation_started` / `generation_finished` を broadcast する。
  - frontend の `loadDocumentPage` で image と explanation を個別に先行反映する。
  - frontend の ws 受信で generation status を flow state に反映する。
  - `WorkspaceView` は image / explanation が片方だけ先に来た場合も表示する。

- テスト
  - frontend に `documentSessionFlow.test.ts` を追加して、image / explanation の先行描画を固定する。
  - `documentSessionState.test.ts` に generation status の開始 / 終了を追加する。
  - backend の `test_generation_task.py` に generation broadcast と cache hit 非通知を追加する。

- 確認
  - `cd /workspace/v2/backend && uv run python -m compileall src`
  - `cd /workspace/v2/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest==8.3.5 --with pygments==2.20.0 pytest tests/test_generation_task.py tests/test_generation_queue.py tests/test_settings.py tests/test_repository.py tests/test_costs.py tests/test_session_sync.py tests/test_voice_utils.py tests/test_voicevox_url.py tests/test_pdf_utils.py`
  - `cd /workspace/v2/frontend && npm test -- --run src/documentSessionFlow.test.ts src/documentSessionState.test.ts src/documentSessionSync.test.ts src/sessionTopPanelState.test.ts src/pageState.test.ts src/audioPreferences.test.ts`
  - `cd /workspace/v2/frontend && npm run build`
