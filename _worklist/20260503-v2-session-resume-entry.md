# 2026-05-03 v2 session resume entry

## 目的
- `続きから` の入口を追加する。
- `documents` に `current_page` を持たせ、続き位置を復元できるようにする。
- backend の session 一覧と snapshot を frontend の開始画面につなぐ。

## 作業内容
- [x] backend repository に `current_page` を追加する。
- [x] backend に `GET /sessions/` と `GET /sessions/{request_id}` を追加する。
- [x] frontend API に session 一覧と snapshot 取得を追加する。
- [x] frontend に開始・続きからの入口画面を追加する。
- [x] `useDocumentSession` に resume 入口を追加する。
- [x] build / test を確認する。

## 完了条件
- `開始` で新規 session を作れる。
- `続きから` で既存 session の current page から再開できる。
- session 一覧に current page が表示される。
- 変更内容が docs と worklist に反映されている。

## 確認
- `cd /workspace/v2/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest==8.3.5 --with pygments==2.20.0 pytest tests/test_repository.py tests/test_voice_utils.py tests/test_voicevox_url.py tests/test_pdf_utils.py`
- `cd /workspace/v2/frontend && npm run build`
