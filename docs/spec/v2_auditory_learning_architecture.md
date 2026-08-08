# v2 実装方針

この文書は、`v2` を `quick` で採っている書き方に寄せるための実装方針をまとめる。
セッションが終わっても参照できるように、コード配置と責務分割を先に固定する。

## 基本方針
- `App.tsx` を太らせない。
- UI は表示に寄せ、状態更新とメッセージ変換は pure function に逃がす。
- 副作用は hook と API レイヤーに閉じる。
- 受信メッセージをその場で直接 UI に反映せず、state 更新関数を経由する。
- 仕様にない fallback は増やさない。

## 役割分担

### Postgres
- session 一覧
- session の現在状態
- 解説文
- お気に入り
- 進行履歴
- 必要なら将来のメモ
- 同期の正本

### WebSocket
- 複数デバイスの動作同期
- session の snapshot と page 更新イベントの配信
- `start` / `resume` / `next` / `stop` / `regenerate` / `playback_started` などの操作通知
- 状態更新イベントの配信
- 接続し直した端末の追従

### ファイル
- PDF 本体
- ページ画像
- 音声ファイル
- 生成キャッシュ

## フロントエンド構成
`quick` の方針に合わせ、`App.tsx` を orchestration 専用にする。

### 推奨ディレクトリ
- `src/api/`
  - HTTP 呼び出し
  - WebSocket の低レベル接続補助
  - payload 型
- `src/commands/`
  - 送信コマンドの生成
  - pure な request builder
- `src/messages/`
  - 受信メッセージの型
  - 受信メッセージから state 更新用 patch を作る関数
- `src/state/`
  - `AppState`
  - `SessionViewState`
  - reducer / apply 関数
- `src/hooks/`
  - `useSessionSocket`
  - `useAudioPlayer`
  - `useBackendDirectoryData`
  - 作用を持つロジック
- `src/components/`
  - 表示専用の部品
  - 状態は props で受ける
- `src/App.tsx`
  - 画面全体の配線
  - hook の接続
  - state の受け渡し
  - ルーティングや表示切り替えの最上位

### App.tsx に置くもの
- hook の呼び出し
- 画面遷移のトリガー
- 送信コマンドの呼び出し
- 受信 patch の適用
- `useDocumentSession` などの画面状態 hook を通じた配線

### App.tsx に置かないもの
- state の細かい更新ロジック
- WS メッセージの分岐処理
- payload 生成の組み立て
- 画面表示に直接関係しない business logic

## state 設計
`quick` と同じく、state は用途で分ける。

### View state
- 現在の session
- 現在の論文
- 検索結果
- 次候補
- 再生状態
- 表示中のメモや補助情報

### App state
- view state に加えて、画面全体の補助状態
- 取得済みの一覧
- エラー表示
- ローディング状態

### 原則
- state 更新は pure function で行う。
- session を切り替えるときは、session 専用 state をまとめてリセットする。
- stale なメッセージは破棄できるようにする。
- 同じイベントを何度受けても壊れない形にする。
- WebSocket の状態遷移は simulator ベースの pure function テストで検証する。

## メッセージ設計
`quick` と同じく、送るデータと受けるデータを分ける。

### 送信
- `buildXxxCommand` のような pure builder を作る。
- UI から直接 object literal をばらまかない。

### 受信
- 受信イベントは handler で分類する。
- handler は「state patch」「更新したい表示」「再取得要否」を返す。
- hook や App は、その結果を適用するだけにする。

## テスト方針
- pure function は単体テストする。
- command builder は期待する JSON だけを検証する。
- message handler は patch と副作用フラグだけを検証する。
- state reducer は入力 state と event から出力 state を検証する。
- API と WebSocket は境界でモックする。

## データの置き方
- Postgres に置くものとファイルに置くものを分ける。
- 画像や音声を DB に入れない。
- document は論文の正本とする。
- session は 1 回の再生・閲覧の単位とする。
- result は document の処理結果であり、同時に session にも属する。
- URL と document の対応は DB に寄せる。
- 説明文は result として DB に寄せる。
- document ごとの既定プロンプトは `prompt_explain.txt` と `prompt_speak.txt` の 2 系統に分ける。
- 解説用プロンプトは `AUDITORY_LEARNING_V2_PROMPT_EXPLAIN_PATH` で指すファイルから読む。相対パスはリポジトリルート基準で解決する。
- 読み上げ用プロンプトは `AUDITORY_LEARNING_V2_PROMPT_SPEAK_PATH` で指すファイルから読む。相対パスはリポジトリルート基準で解決する。
- session ごとの prompt / model 設定は DB に寄せる。解説用プロンプトと読み上げ用プロンプトは別々に保持する。
- 既定のモデル名は `AUDITORY_LEARNING_V2_DEFAULT_MODEL_NAME` で与える。既定値は `gpt-5.6-luna` とする。
- 既定の reasoning effort は `AUDITORY_LEARNING_V2_DEFAULT_REASONING_EFFORT` で与える。既定値は `medium` とする。
- 生成物は `_data` / `_cache` に置く。
- document メタデータは `papers`、session 状態は `sessions`、処理結果は `session_results` に閉じ込める。
- 利用情報は `session_usage_records` に append-only で積む。
- session には `total_generation_count`、`total_generation_elapsed_ms`、`total_input_tokens`、`total_output_tokens`、`total_cost_usd` を持たせる。
- 実装は Postgres を主系にする。テストでは repository を差し替えて扱う。
- session 一覧は `sessions` 系の正本から返す API を用意し、resume の入口にする。
- `sessions` には少なくとも `current_page` と `page_num` を持たせ、一覧と snapshot から続き位置を復元できるようにする。
- `GET /sessions/` は一覧、`GET /sessions/{request_id}` は snapshot として扱う。
- `GET /favorites/` は favorite 一覧、`POST /favorites/{request_id}/toggle` は favorite 切り替えとして扱う。favorite の保存単位は session と page の組で、`request_id` で session を特定し、`page_num` があればそれを、なければその時点の `current_page` を対象にする。
- `GET /sessions/{request_id}/settings` と `PATCH /sessions/{request_id}/settings` で解説用プロンプト、読み上げ用プロンプト、model を読む・更新する。
- `POST /sessions/{request_id}/favorite` は favorite toggle の session 版エイリアスとして扱う。
- `session_results` は session ごとの処理結果の最新状態を持つ。1 つの result は 1 つの paper と 1 つの session に属する。
- `session_usage_records` は生成ごとの利用情報を持つ。result は上書きしても usage は上書きせず、常に追加する。
- generation 開始時は `generation_started`、完了時は `generation_finished` を ws で配信し、ヘッダーに小さく生成中表示を出す。
- 解説とプレビューは独立して先行描画できる場合は先に描画し、読み込み中はぐるぐるで待機状態を表す。
- プレビューはマウスホイールで拡大縮小できる。
- 生成キューは優先度付きで扱い、現在の再生対象が予約された場合は同じ `task_id` の既存予約を高優先度に更新する。
- `explain` 系は生成済みキャッシュがあっても一度キューを通し、worker が実行直前にキャッシュを確認する。
- `regenerate` 系はキャッシュがあっても再生成する。

## 画面構成
- `開始` は start page に置く。
- start page では PDF URL の入力に加えて、PDF ファイルのアップロード開始もできる。
- `続きから` は start page に置く。
- `お気に入り` は独立ページとして置く。
- `session 一覧`
- `現在の session`
- `現在の session` では URL は表示するだけにし、URL 入力と開始操作は置かない。
- `現在の session` では URL を変更でき、その URL を再生する操作を置く。
- `詳細` の折りたたみには prompt と model を置き、保存ボタンの下に読み上げ文を表示し、session の累積統計とコストも表示する。
- `開始・続きから` の一覧は polling で更新し、手動更新ボタンは置かない。

`quick` と同じく、一覧は再開導線として扱う。
複数端末同期を使う場合は、一覧と current session を同じ状態体系で扱う。
`start` と `session` は別ページとして分け、session 開始後は current session page へ遷移する。
