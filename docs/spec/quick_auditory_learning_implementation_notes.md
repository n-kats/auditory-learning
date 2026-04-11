# quick-auditory-learning 実装上の注意

この文書は、仕様そのものではなく、実装時に外しやすい前提と注意点をまとめたもの。
仕様の正本は `quick_auditory_learning.md` / `quick_auditory_learning_messages.md` / `quick_auditory_learning_flow.md` を参照する。

## まず守ること
- 仕様に書いていない fallback は実装しない。
- `session_started` は session の確立であり、再生可能状態の確立ではない。
- 再生可能状態は `paper_ready` で決まる。
- `paper_ready` と `paper_search_updated` の順序は前後してよい。

## セッション開始時の表示
- `currentSessionId` があるのに `currentPaper` がまだ無い間は、start 文言ではなく loading を維持する。
- セッションに入った直後に自動再生しない。
- `paper_ready` が来るまで、検索結果セクションは出さない。

## 続きから一覧の表示
- `session.status` は backend の内部状態として保持してよい。
- ただし `続きから` の一覧には `active` / `stopped` を表示しない。
- 一覧に出すのは session_id、更新時刻、処理時間、費用などの必要最小限にする。

## 接続数の表示
- `sessions/recent` は session ごとの WebSocket 接続数を返してよい。
- UI は `続きから` 一覧で title を 1 行で表示し、session ID、処理時間、接続数はまとめて別行に出してよい。
- その別行では、処理時間を session ID と接続数の間に置く。
- 接続数が 0 のときは表示しない。
- `WebSocket` の接続状態そのものとは別に扱う。

## セッション情報の表示
- `続きから` 一覧には current paper の title を表示してよい。
- title は session ID と同じ文字サイズ・色に揃える。
- 長い title は 1 行で省略してよい。

## WebSocket の扱い
- WebSocket のメッセージは、受信順どおりに直列で処理する。
- `session_id` が違うイベントは、その session の state に混ぜない。
- `paper_id` が違う `paper_search_updated` は stale として捨てる。
- `paper_ready` と `paper_search_updated` が逆順でも、最終的に正しい paper に紐づく状態だけを残す。
- 同じ session を複数のクライアントで開いているときは、`next` / `set_next_candidate` / `paper_ready` / `session_advanced` の結果を全クライアントで共有して反映する。
- 同じ session を複数のクライアントで開いているときは、`regenerate` も全クライアントで共有する。
- 片方が停止中や待機中でも、別クライアントで session が進んだら、再生再開時には current paper を最新状態へ追従させる。
- 同期の詳細ルールは `quick_auditory_learning_sync_policy.md` に従う。

## 検索結果の扱い
- `search` の `rejected_candidates` は、その paper の検索で hits に入らなかった候補を表す。
- 画面の `検索結果` は current paper の結果を出す。
- 画面の `前の論文から検索した他の論文` は、直前 paper の `rejected_candidates` を引き継いで見せる。
- `はじめから` や `停止` では、その引き継ぎ状態を破棄する。

## 次候補の扱い
- `next_paper_id` は現在の次候補を表す単一値である。
- `queue` / `dequeue` は FIFO の待ち行列ではなく、次候補の指定・変更として扱う。
- `set_next_candidate` は候補を指定する操作であり、解除トグルではない。
- UI の選択状態は backend の `next_paper_id` を基準にする。
- UI は `set_next_candidate` の送信直後に候補行を楽観的に選択表示してよい。backend の `session_next_candidate_updated` で同期する。

## 先読みの扱い
- 次候補の検索先読みは、`session_playback_started` を受けてから始める。
- `next_paper_id` が決まっただけでは先読みを始めない。
- 先読みが走り切っても、ターゲットが変わっていたら cache に入れない。
- 先読み結果は `(session_id, paper_id)` が一致したものだけ消費する。

## 通信メッセージの注意
- `session_started` で session の基本状態を整える。
- `paper_ready` で再生に必要な情報を反映する。
- `paper_search_updated` は後追いで検索結果だけを更新する。
- `session_next_candidate_updated` は次候補の更新通知であり、検索結果更新とは別物。

## 変更時の確認
- セッション開始、再開、停止の各経路で、検索結果が残りすぎないことを確認する。
- `paper_ready` / `paper_search_updated` / `session_next_candidate_updated` の順序差分をテストで固定する。
- `next_paper_id` や検索結果の扱いを変える場合は、frontend の状態遷移テストと backend の回帰テストを同時に更新する。
