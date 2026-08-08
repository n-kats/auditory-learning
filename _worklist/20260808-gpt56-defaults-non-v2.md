# v2 以外の GPT-5.6 Luna 既定値対応

## 目的

- root の `auditory_learning` と `quick-auditory-learning` でも GPT-5.6 系を利用できるようにする。
- 既定モデルを `gpt-5.6-luna`、既定 reasoning effort を `medium` に揃える。

## 対応内容

- [x] root アプリのモデル、reasoning effort、README の記載を更新する。
- [x] quick アプリのモデル設定、reasoning effort、Compose、README、料金表を更新する。
- [x] quick の既存テストを実行する。

## 確認結果

- `uv run --project quick-auditory-learning/backend pytest tests/test_quick_auditory_learning.py` は 71 件成功した。
- `uv run --project . pytest tests/test_pdf_utils.py` は 2 件成功した。
- 両アプリの `compileall` と `git diff --check` は成功した。
- quick の Ruff は既存の import 順序、未使用 import、長行の指摘で失敗した。今回の変更箇所に限定した修正は行っていない。
