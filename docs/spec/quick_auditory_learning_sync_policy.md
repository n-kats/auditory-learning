# quick-auditory-learning 同期ポリシー

この文書は、同一 session を複数クライアントで開いたときの同期方針と、検証テストの観点をまとめる。

## 基本方針
- backend を session 状態の唯一の正とする。
- 同じ session を開いているクライアントは、同じ session room の更新を共有する。
- 片方のクライアントだけで進んだ操作でも、同じ session に属する他クライアントへ反映される。
- 逆に、`resume` や HTTP snapshot replay のような履歴再生は、そのクライアント自身の復元に限定する。他クライアントへ過去イベントを再送しない。
- HTTP snapshot replay のあとに送る `resume` も、過去イベントの再送ではなく session room 参加と差分受信のために使う。

## 共有する更新
- `next`
- `set_next_candidate`
- `stop`
- `regenerate`
- `playback_started`
- それらに続く `session_next_requested` / `session_advanced` / `session_next_candidate_updated` / `session_regenerated` / `session_stopped` / `session_playback_started`
- `paper_ready`
- `paper_search_updated`
- `session_costs_updated`

## 局所的な再生
- `resume` は socket 再接続時の差分 replay であり、履歴を読むクライアントだけに返す。
- `start` は新規 session 作成の初期応答であり、他クライアントへ旧履歴を巻き戻して送らない。
- 前面が停止中でも、別クライアントが session を進めていれば、再生再開時には最新の current paper に追従する。

## 検証テスト
- 2 つの websocket クライアントで同じ session を開き、片方の `next` がもう片方にも `session_advanced` と `paper_ready` を配信することを確認する。
- 同じ条件で `regenerate` が両方へ配信されることを確認する。
- `resume` の replay が他クライアントへ過去イベントを漏らさないことを確認する。
- 再生停止中のクライアントが再開したとき、最新の current paper を反映することを確認する。
