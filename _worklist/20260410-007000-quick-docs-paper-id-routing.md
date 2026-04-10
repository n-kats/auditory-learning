# 20260410 quick-auditory-learning paper_id routing docs

## 目的
`source_url` のときだけ arXiv API を使い、`paper_id` 系では DB 優先で arXiv fallback しない仕様を docs に反映する。

## 完了
- `docs/spec/quick_auditory_learning.md` を追加した。
- `docs/spec/仕様.md` に quick 仕様の参照を追記した。

## 確認
- 実装と docs の記述が一致していることを確認する。
- `paper_id` 系では arXiv API を呼ばないことをテストで固定済み。

## 次
- もし route / memo / audio の仕様変更が出たら、この仕様書を先に更新する。
