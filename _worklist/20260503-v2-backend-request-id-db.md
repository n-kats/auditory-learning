# 20260503 v2 backend request id db

## 目的
- v2 backend の request_id 生成と document メタデータを Postgres repository に移す。
- 将来の session / explanation 永続化に備えて、backend の状態保存層を分離する。

## 作業項目
- [x] backend 設定を `settings.py` に切り出す。
- [x] `request_id` の対応表を Postgres repository に移す。
- [x] document メタデータを repository に保存する。
- [x] `psycopg` 依存を追加する。
- [x] `docker-compose.yml` に Postgres サービスを追加する。
- [x] session 一覧 API の入口を追加する。
- [x] repository のテストを追加する。
- [x] `uv lock` で lockfile を再生成する。
- [x] `uv sync` で backend 依存を反映する。
- [x] build とテストを確認する。

## 確認手順
- [x] `cd /workspace/v2/backend && python -m compileall src`
- [x] `cd /workspace/v2/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest==8.3.5 --with pygments==2.20.0 pytest tests/test_repository.py tests/test_voice_utils.py tests/test_voicevox_url.py tests/test_pdf_utils.py`
