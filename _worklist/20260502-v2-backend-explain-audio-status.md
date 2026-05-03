# 20260502 v2 backend explain audio status

## 目的
- `/explain/` と `/regenerate/` のレスポンスに音声生成状態を追加する。
- 音声生成が失敗しても解説は返し、その失敗理由を UI で扱えるようにする。

## 作業項目
- [x] 生成キューの返却値に音声状態を含める。
- [x] `ExplainResponse` に音声状態フィールドを追加する。
- [x] フロントの `ExplainResponse` 型を合わせる。
- [x] バックエンドのビルド/テストで確認する。

## 確認手順
- [ ] `cd /workspace/v2/backend && python -m compileall src`
- [ ] `cd /workspace/v2/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest tests/test_voice_utils.py tests/test_voicevox_url.py tests/test_pdf_utils.py`
