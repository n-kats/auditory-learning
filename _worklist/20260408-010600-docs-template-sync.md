# docs テンプレート同期

## TODO
- [x] `docs/directory_structure.md` を `_local/templates/template_AGENTS.md` の `docs` 運用に合わせて更新する。
- [x] `docs/spec/`, `docs/how_to/`, `docs/coding_rule/`, `docs/web/` を作成する。
- [x] 既存の `docs/workflows/` や `docs/images/` と矛盾しないか確認する。

## 決定事項
- 既存の `docs/` 配下は残しつつ、テンプレートで想定されている `docs` の置き場を追加する。
- `docs/spec/`、`docs/how_to/`、`docs/coding_rule/`、`docs/web/` には、それぞれ `仕様.md`、`手順.md`、`規約.md`、`参照ログ.md` を置く。

## 未決
- なし。

## 確認手順
- `docs/directory_structure.md` に新しい `docs` 配下の置き場が載っていることを確認する。
- `rg --files docs` で `docs/spec/`, `docs/how_to/`, `docs/coding_rule/`, `docs/web/` が存在することを確認する。
