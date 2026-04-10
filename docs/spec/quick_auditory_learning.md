# quick-auditory-learning 仕様

## 目的
公開された arXiv 論文の URL を起点に、論文の解説文と音声を生成し、再生セッションとして追跡する。

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
- その境界より前に処理が完全終了していた分は、先読みとして除外し、`時間（先読み除く）` を `0` にできる。
- 未完了の行は `pending` とし、`0` に潰さない。
- `prefetch` は内部計測として残してよいが、表示上は通常の行と分けて扱う。

## コスト
- 画面に表示するコストの行は、`search` / `embedding` / `explanation` / `audio` / `keyword_generation` / `query_generation` のみを表示する。
- 各コスト行は、処理の開始時刻 `started_at` と終了時刻 `finished_at` を記録する。
- 合計の時間は、同じセッションや同じ論文に対する記録済みの生成区間を重ね合わせた壁時計時間とする。個別の処理時間の単純加算ではない。
- 合計の費用は、記録済みの生成コストをすべて反映する。途中で止まった処理でも、完了して記録された分は含める。
- `prefetch` は内部計測として残してよいが、画面の行としては表示しない。
- 未計算のコスト行は `0` ではなく `pending` として返す。
- 各コスト行は、通常の `時間` に加えて `処理待ち時間` を持つ。
- 合計行も同じ列構成にして、`合計` と `処理待ち時間` を並べて表示する。
- `処理待ち時間` は、`next` が backend に受理された時刻から各処理の `finished_at` までを計算する。
- その時刻より前に処理が完全終了していた場合は、`処理待ち時間` を `0` とする。
- 合計値は、記録済みの rows からその時点の集計値を返す。未完了の行だけが `pending` になる。
- 先読みで実際に生成した解説と音声の費用は、通常の `explanation` / `audio` として計上する。
- セッション一覧とコスト詳細の合計は、記録済みの rows から再計算した値を使う。

## セッション
- `next_paper_id` は backend が決定する。
- frontend は第一候補を独自に計算しない。
- `next` 要求の受付時刻は `session_next_requested` として記録する。
- `session_queued` と `session_costs_updated` は session WebSocket で配信する。

## テスト方針
- route の path converter はテストで固定する。
- `paper_id` 系の処理が arXiv API を呼ばないことを unit test で確認する。
- route や同期処理のテストは実 DB ではなく `tmp_path` とモックで行う。
