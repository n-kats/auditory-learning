# 20260411 quick-auditory-learning hide session status in list

## 目的
「続きから」の一覧で `session_id / active` のような表示を出さないようにする。

## 方針
- backend の `status` は保持してよい。
- ただし UI の一覧には出さない。
- 将来の再発を避けるため、表示用 helper を通すか、少なくとも描画箇所を一か所に閉じる。

## 確認
- 変更後に frontend のテストと typecheck を通す。

## 完了
- `続きから` 一覧の `session {session_id} / active` 表示を `session {session_id}` に変更した。
- `docs/spec/quick_auditory_learning_implementation_notes.md` に、一覧に session status を出さない注意を追記した。

## 追加
- session WebSocket 接続数を `/sessions/recent` から取得し、`続きから` の各 session 行で `session {session_id}` の近くに `接続数: N` と表示するようにした。
- 接続数が 0 のときは表示しないようにした。
- current paper の title は 1 行で独立表示し、それ以外の session 情報は別行にまとめるようにした。
- current paper の title は session ID と同じ文字サイズ・色に揃えた。

## 仕様追記
- `docs/spec/quick_auditory_learning.md` に `続きから` 一覧の表示ルールを追記した。
- `docs/spec/quick_auditory_learning_implementation_notes.md` に title の文字サイズ・色の揃え方を追記した。
- 同じ session を複数クライアントで開いたとき、`next` や `set_next_candidate` の結果が全クライアントへ反映されることを追記した。
- 同じ session を複数クライアントで開いたとき、`regenerate` も共有されることと、停止中のクライアントは再生再開時に最新 current paper に追従することを追記した。
- `docs/spec/quick_auditory_learning_sync_policy.md` を追加し、同期方針と検証テストの観点を整理した。

## 検証
- backend では、共有すべき session コマンドの判定を helper 化し、単体テストで固定する。
- frontend では、後続の `paper_ready` で current paper が差し替わることを回帰テストで固定する。
