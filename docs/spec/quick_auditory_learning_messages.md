# quick-auditory-learning 通信データ仕様

## 目的
quick-auditory-learning でやり取りする通信データの種類を、HTTP と WebSocket に分けて整理する。

## セッション種別
backend が永続化する session の状態は次の 2 種類だけを扱う。

- `active`: 進行中の session
- `stopped`: 停止済みの session

補足:
- ここでいう session 種別は backend の永続状態を指す。
- frontend の `loading`、WebSocket の `connected` などは通信状態であり、session 種別ではない。

## 通信の入口
### HTTP
単発取得、一覧取得、メモやお気に入りなどの取得・更新に使う。

### session WebSocket
`/sessions/ws` を使う。session の開始、再開、次へ、停止、再生成、次候補指定、状態更新通知をやり取りする。

### memo WebSocket
`/papers/{paper_id}/memo/ws` を使う。論文メモの初期値と更新をやり取りする。

## HTTP の通信データ型
### 取得系
| エンドポイント | 入力 | 出力 | 用途 |
| --- | --- | --- | --- |
| `GET /health` | なし | `HealthResponse` | backend 生存確認 |
| `GET /config` | なし | 設定 JSON | 画面初期化 |
| `GET /embedding-models?model_name=...` | `model_name` | `EmbeddingModel[]` | 埋め込みモデル一覧 |
| `POST /search` | `SearchRequest` | `SearchResponse` | 単発検索 |
| `GET /favorites` | なし | `FavoriteListResponse` | お気に入り一覧 |
| `GET /history/recent` | `limit` | transition 配列 | 再生履歴 |
| `GET /sessions/recent` | `limit` | `SessionListResponse` | セッション一覧 |
| `GET /sessions/{session_id}` | なし | `SessionSnapshot` | session 再開の snapshot |
| `GET /sessions/{session_id}/events` | `after_seq` | `SessionEvent[]` | session イベント再生 |
| `POST /papers/resolve` | `PaperResolveRequest` | `PaperResolveResponse` | source URL から論文解決 |
| `GET /papers/{paper_id}/memo` | なし | `PaperMemoResponse` | メモ初期値 |
| `POST /explanations/{paper_id}` | なし | `ExplanationResponse` | 解説と音声生成 |
| `GET /audio/{paper_id}` | なし | WAV ファイル | 音声本体 |
| `GET /audio/{paper_id}/chunks/{chunk_index}` | なし | WAV チャンク | 音声チャンク |

### 更新系
| エンドポイント | 入力 | 出力 | 用途 |
| --- | --- | --- | --- |
| `POST /favorites/{paper_id}/toggle` | なし | `FavoriteToggleResponse` | お気に入り切替 |
| `POST /history/transition` | `HistoryTransition` | transition JSON | 履歴記録 |
| `PUT /papers/{paper_id}/memo` | `PaperMemoUpdateRequest` | `PaperMemoResponse` | メモ保存 |

## session WebSocket のクライアントコマンド
`SessionClientMessage.type` に入る値を列挙する。

| type | 必須入力 | 用途 | 補足 |
| --- | --- | --- | --- |
| `start` | `source_url` | 新しい session を開始 | `model_name`, `limit`, `route1_weight`, `route2_weight`, `seed`, `search_modes` も送る |
| `resume` | `session_id`, `last_event_seq` | websocket 再接続時の差分再開 | 現在の「続きから」ボタンは HTTP snapshot replay を使うが、socket 再接続ではこのコマンドを使う |
| `next` | `session_id` | 次の論文へ進む | next candidate があればそれを優先する |
| `set_next_candidate` | `session_id`, `paper_id` | 次に再生する候補を指定する | 明示的な次候補指定。別候補を指定すると更新される。 |
| `stop` | `session_id` | session を停止 | current session を `stopped` にする |
| `regenerate` | `session_id` | 現在の論文を再生成 | 解説・音声を作り直す |
| `playback_started` | `session_id`, `paper_id` | 再生が実際に始まったことを backend に知らせる | 再生開始後に次論文の検索先読みを始めるための合図 |

## session WebSocket のサーバーイベント
`SessionEventMessage.type` に入る値を列挙する。

| type | 送信タイミング | 主な payload |
| --- | --- | --- |
| `session_started` | session 作成直後 | `session_id`, `source_url`, `config`, `root_paper`, `origin` |
| `session_playback_started` | audio 再生が始まった時 | `session_id`, `paper_id` |
| `paper_ready` | 再生準備ができた時 | `session_id`, `paper`, `origin`, `from_paper_id`, `trail_paper_ids`, `next_paper_id`, `search_deferred`, `search`, `explanation`, `audio_url`, `audio_urls`, `audio_duration_ms`, `paper_costs`, `session_costs`, `memo`, `notices` |
| `paper_search_updated` | 検索結果が後から更新された時 | `session_id`, `paper_id`, `origin`, `from_paper_id`, `next_paper_id`, `search`, `notices` |
| `session_next_candidate_updated` | 次候補が指定・更新された時 | `session_id`, `paper_id`, `next_paper_id` |
| `session_next_requested` | `next` が受理された時 | `session_id`, `from_paper_id`, `to_paper_id` |
| `session_advanced` | current paper が切り替わった時 | `session_id`, `from_paper_id`, `to_paper_id` |
| `session_regenerated` | 再生成が始まった時 | `session_id`, `paper_id`, `title` |
| `session_stopped` | session 停止時 | `session_id`, `status` |
| `session_costs_updated` | コスト確定や更新のたび | `session_id`, `session_costs`, `paper_id` 省略可, `paper_costs` 省略可 |
| `error` | コマンド失敗時 | `session_id`, `message` |

## memo WebSocket の通信データ型
`/papers/{paper_id}/memo/ws` は初回に `PaperMemoResponse` と同形の snapshot を送り、その後も更新時は同じ形を送る。

| type 相当 | 実体 | 用途 |
| --- | --- | --- |
| 初回 snapshot | `PaperMemoResponse` | 現在保存されているメモを返す |
| 更新通知 | `PaperMemoResponse` | `PUT /papers/{paper_id}/memo` の結果を配信する |

## 共通データ型
| 型名 | 用途 |
| --- | --- |
| `Paper` | 論文本文の共通データ |
| `SearchRequest` | 検索要求 |
| `SearchResponse` | 検索結果 |
| `SearchHit` | 検索ヒット |
| `SearchCandidate` | 検索候補 |
| `ExplanationResponse` | 解説と音声 URL |
| `SessionSnapshot` | session の現在状態 |
| `SessionListItem` / `SessionListResponse` | session 一覧 |
| `SessionEvent` | session イベント履歴 |
| `SessionCostsResponse` | session / paper のコスト詳細 |
| `SessionCostItem` | コスト行 |
| `FavoritePaperItem` | お気に入り一覧の 1 行 |
| `FavoriteToggleResponse` | お気に入り切替結果 |
| `PaperMemoResponse` | メモ本文と更新日時 |

## 実装上の注意
- `paper_ready` は再生開始に必要な情報を返す。
- `paper_search_updated` は検索結果だけを後から更新する。
- `paper_ready` と `paper_search_updated` の到着順は前後してよい。
- 同じ session / paper に紐づかない `paper_search_updated` は frontend で破棄する。
- `前の論文から検索した他の論文` は、frontend が直前 paper の `search.rejected_candidates` を保持して表示する UI 領域であり、通信プロトコル上の別イベントではない。
- `session_started` は session 確立を表すだけで、再生可能状態はまだ保証しない。
- `session_playback_started` は、次論文の検索先読みを始めてよい合図として扱う。
- `session_costs_updated` は複数回届く前提で扱う。
