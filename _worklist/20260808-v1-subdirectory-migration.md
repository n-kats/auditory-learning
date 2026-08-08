# root 版を v1 サブディレクトリへ移動

## 目的

- root にある従来版 PDF 解説アプリを `v1/` 配下へ整理する。
- v1 の実行時データを `_data/v1/` に分離する。
- v1 と v2、quick の起動・テスト・ドキュメントの入口を明確に分ける。

## 対応内容

- [x] v1 の backend、frontend、Docker、Python 設定、専用テストを `v1/` へ移動する。
- [x] `prompt.txt` を root の `prompt_for_v1.txt` に変更し、v1 から参照する。
- [x] v1 の出力先を実行時データ置き場である `_data/v1/` に変更する。既存データは削除・移動しない。
- [x] v1 用スクリプトを `scripts/launch_v1.sh` と `scripts/serve_v1.sh` にする。
- [x] 旧 `scripts/launch.sh` と `scripts/serve.sh` は置かず、v1 用スクリプト名に統一する。
- [x] Makefile、root README、v1 README、`docs/directory_structure.md` の責務とパスを更新する。

## 確認手順

- [x] v1 backend のテストと Python 構文チェックを実行する。
- [x] v1 frontend の build を実行する。
- [x] quick と v2 の既存テストを実行する。
- [x] 新旧スクリプトの構文、v1 の参照先、`_data/v1/` への出力、差分形式を確認する。

## 補足

- 既存コードのフォーマット、lint、mypy 指摘は今回の対象外とする。
- `make test` は v1 2 件、quick 71 件、v2 30 件が成功した。
- Docker CLI による起動確認は環境に Docker がないため未実施。
