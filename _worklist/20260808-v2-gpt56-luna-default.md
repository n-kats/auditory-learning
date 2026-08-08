# GPT-5.6 Luna を既定モデルにする

## 目的

- GPT-5.6 系モデルの設定と料金を追加する。
- v2 の既定モデルを `gpt-5.6-luna`、既定 reasoning effort を `medium` にする。

## 対応内容

- [x] backend 設定、DB の新規行用既定値、Docker Compose の環境変数を更新する。
- [x] GPT-5.6 系の料金計算を追加する。
- [x] README と仕様書を更新する。
- [x] backend の関連テストを実行する。

## 確認結果

- `uv run --project v2/backend pytest v2/backend/tests` は 30 件成功した。
- `git diff --check` は成功した。
- Docker CLI が利用できないため、Compose の実行時検証は未実施。
- `gpt-5.6-luna` のコスト計算が入力 $0.20 / 出力 $1.20 であることを確認する。
