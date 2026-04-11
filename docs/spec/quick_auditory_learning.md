# quick-auditory-learning 仕様

## 目的
公開された arXiv 論文の URL を起点に、論文の解説文と音声を生成し、再生セッションとして追跡する。

## 関連文書
- `quick_auditory_learning_messages.md`: session の種類、通信データ型、HTTP / WebSocket の一覧。
- `quick_auditory_learning_flow.md`: start / resume / next / 次候補指定 / stop / regenerate の通信フロー。
- `quick_auditory_learning_implementation_notes.md`: 実装時に踏みやすい注意点。
- `quick_auditory_learning_sync_policy.md`: 同一 session を複数クライアントで開いたときの同期ポリシー。

## 論文取得の方針
- `source_url` から開始する処理だけが arXiv API を利用する。
- `paper_id` を直接扱う処理は、DB からの取得を優先し、arXiv API にはフォールバックしない。
- `paper_id` が DB に無い場合は、保存済みデータが無いものとして `404` を返す。
- 検索の候補が十分に見つからない場合の代替は、DB 内の論文からランダムに選ぶことに限る。
- DB 外の論文を取るためのランダム fallback は使わない。

## フォールバック方針
- フォールバックは基本的に導入しない。
- 仕様に書いていない fallback は追加しない。

## URL と route の扱い
- `source_url` は `resolve_paper_from_source()` で解決する。
- `paper_id` は `resolve_paper_from_identifier()` 相当の DB 優先処理で扱う。
- `paper_id` に `/` を含む場合があるため、FastAPI の path converter を使う。
- `audio` 系は `audio_chunk` を `audio` より先に定義し、`/audio/{paper_id:path}/chunks/{chunk_index}` が `/audio/{paper_id:path}` に吸われないようにする。

## メモ
- メモの取得、更新、WebSocket 通知は `paper_id` 単位で行う。
- メモの初期値は HTTP で取得し、更新通知は WebSocket で受ける。

## お気に入り
- お気に入り一覧は `paper_id` と `title` を返す。
- frontend は `title` を表示し、`paper_id` は補助情報として扱う。

## 音声生成
- 音声は解説文から生成する。
- `audio` / `audio_chunk` は、保存済みの音声ファイルがあればそれを返す。
- 保存済み音声が無ければ、まず DB の解説文を確認し、無ければ解説を生成してから音声を作る。
- `paper_id` 系の音声取得で arXiv API を再検索してはいけない。

## 計測指針
- 生成コストは `created_at` だけで近似せず、各処理の `started_at` と `finished_at` を記録する。
- 先読み除く集計の境界は、`next` が backend に受理された時刻とする。
- UI 上の列名は `処理待ち時間` とする。
- `処理待ち時間` は、各処理について `next` の時刻から `finished_at` までを計算する。
- その境界より前に処理が完全終了していた分は、先読みとして除外し、`処理待ち時間` を `0` にできる。
- 未完了の行は `pending` とし、`0` に潰さない。
- `prefetch` は内部計測として残してよいが、表示上は通常の行と分けて扱う。

## コスト
- 画面に表示するコストの行は、`search` / `embedding` / `explanation` / `audio` / `keyword_generation` / `query_generation` のみを表示する。
- 各コスト行は、処理の開始時刻 `started_at` と終了時刻 `finished_at` を記録する。
- 全体の時間は、同じセッションや同じ論文に対する記録済みの生成区間を重ね合わせた壁時計時間とする。個別の処理時間の単純加算ではない。
- 全体の費用は、記録済みの生成コストをすべて反映する。途中で止まった処理でも、完了して記録された分は含める。
- `prefetch` は内部計測として残してよいが、画面の行としては表示しない。
- 未計算のコスト行は `0` ではなく `pending` として返す。
- 各コスト行は、通常の `時間` に加えて `処理待ち時間` を持つ。
- 全体行も同じ列構成にして、`全体` と `処理待ち時間` を並べて表示する。
- `処理待ち時間` は、`next` が backend に受理された時刻から各処理の `finished_at` までを計算する。
- その時刻より前に処理が完全終了していた場合は、`処理待ち時間` を `0` とする。
- 全体値は、記録済みの rows からその時点の集計値を返す。未完了の行だけが `pending` になる。
- 先読みで実際に生成した解説と音声の費用は、通常の `explanation` / `audio` として計上する。
- セッション一覧とコスト詳細の全体は、記録済みの rows から再計算した値を使う。

## セッション
- セッションに入った直後は自動再生しない。再生は明示的な再生操作で開始する。
- `session_started` は session の確立を表し、再生可能状態の確立は `paper_ready` で扱う。
- `session_started` 直後で `currentSessionId` が存在し `currentPaper` がまだ無い間は、start 文言ではなく loading 表示を維持する。
- `next_paper_id` は backend が決定する。
- frontend は第一候補を独自に計算しない。
- `set_next_candidate` は次に再生する候補を指定する操作である。`next_paper_id` はその結果として backend が決める。
- `set_next_candidate` で選ばれた候補は、検索結果一覧の候補行で色付き表示する。
- UI は `set_next_candidate` の送信直後に候補行を楽観的に色付き表示してよい。backend の `session_next_candidate_updated` で同期する。
- `next_paper_id` が決まっても、再生開始前は検索先読みを始めない。
- `session_playback_started` を受けてから、その時点の current paper に紐づく next_paper_id の検索先読みを始める。
- `next` 要求の受付時刻は `session_next_requested` として記録する。
- `paper_ready` は再生開始に必要な情報を返し、検索結果は後続の `paper_search_updated` で反映する。
- 検索結果の主表示は current paper に紐づく `session_id` と `paper_id` の結果である。frontend は別セッションや別論文の検索結果を流用せず、論文が切り替わったら current paper の検索結果を切り替える。
- `前の論文から検索した他の論文` は、直前の論文で得られた `rejected_candidates` を次の論文表示へ引き継いで見せる UI 領域である。はじめから開始した場合や停止後はこの引き継ぎ状態を破棄する。
- `paper_search_updated` は、対応する論文の検索結果が後から届くイベントであり、`paper_ready` より前後どちらに届いてもよい。
- `session_next_candidate_updated` と `session_costs_updated` は session WebSocket で配信する。
- 同じ session を複数のクライアントで開いている場合、`next` や `set_next_candidate` の結果は同じ session の全クライアントに反映される。
- 同じ session を複数のクライアントで開いている場合、`regenerate` の結果も同じ session の全クライアントに反映される。
- 片方のクライアントが再生停止や待機状態にあっても、別クライアントが session を進めたら、再生再開時には current paper を最新状態へ追従させる。

## 続きから一覧の表示
- `続きから` の一覧では current paper の title を 1 行で独立表示する。
- その title は session ID と同じ文字サイズ・色に揃える。
- title 以外の session 情報は別行にまとめる。
- 接続数は `接続数:` として表示し、0 のときは出さない。

## テスト方針
- route の path converter はテストで固定する。
- `paper_id` 系の処理が arXiv API を呼ばないことを unit test で確認する。
- route や同期処理のテストは実 DB ではなく `tmp_path` とモックで行う。
