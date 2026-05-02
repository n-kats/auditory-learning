# 2026-04-21 OpenReview PDF 取得 403 対策

- [x] `auditory_learning/server.py` の PDF ダウンロードをブラウザ相当のヘッダ付きに変更する。
- [x] OpenReview 相当の URL で `httpx.get` に渡すヘッダと `follow_redirects` を検証するテストを追加する。
- [x] 必要なら README に取得制約と対策を追記する。
- [x] テストを実行して結果を記録する。

## 結果
- `auditory_learning/utils/pdf_utils.py` を追加し、`User-Agent` と `Accept` を付けた `httpx.get(..., follow_redirects=True)` で PDF を取得するようにした。
- `tests/test_pdf_utils.py` でヘッダとリダイレクト追従、HTTP エラーの伝播を検証した。
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q /workspace/tests/test_pdf_utils.py` は 2 件成功。
