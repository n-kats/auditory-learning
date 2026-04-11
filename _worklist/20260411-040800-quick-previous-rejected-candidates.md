# quick-auditory-learning 前論文候補の表示修正

## TODO
- [x] 現在の論文の rejected_candidates と、前の論文から引き継いだ rejected_candidates を分離する。
- [x] `前の論文から検索した他の論文` は、直前の論文の rejected_candidates を表示するようにする。
- [x] はじめから開始したときは、前セッションの候補を残さない。
- [x] 仕様文とテストを新しい表示方針に合わせて更新する。
- [x] 画面の `前の論文から検索した他の論文` は previous rejected_candidates を使うように修正する。
- [x] 古い候補の prefetch が完了しても、決定済み paper_id と一致しない場合は cache に入れず送信しない。

## 方針
- current paper の検索結果と previous paper の検索結果を別 state に分ける。
- `paper_ready` で current paper が切り替わるとき、直前 paper の rejected_candidates を previous 側へ退避する。
- loading 中は従来どおり検索結果セクションを出さない。
- `前の論文から検索した他の論文` は、直前 paper の rejected_candidates を引き継いで表示する。

## 確認
- `cd /workspace/quick-auditory-learning/frontend && npm test`
- `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
- `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`
