# v2 auditory learning

公開 PDF の URL または PDF ファイルから読み始め、PDF のプレビューとページごとの AI 解説を見ながら、VOICEVOX で音声再生するアプリです。

開始画面では新しい PDF の開始、以前に読んだ PDF の「続きから」の再開、お気に入りの確認ができます。再生画面では解説と PDF のプレビューを並べて表示し、再生・一時停止、音量・速度の変更、ページ移動、再生成、お気に入り登録を操作できます。解説用プロンプト、読み上げ用プロンプト、モデル、Reasoning Effort も調整できます。

## 構成

- `backend/`: FastAPI 本体
- `frontend/`: React + Vite の UI
- `postgres/`: backend の永続データを持つ Postgres
- `voicevox/`: 音声合成エンジン
- `docker-compose.yml`: backend, frontend, postgres, voicevox の起動定義

## 起動

```bash
bash scripts/launch_v2.sh
```

停止:

```bash
bash scripts/down_v2.sh
```

## 環境変数

- `OPENAI_API_KEY`
- `AUDITORY_LEARNING_V2_DATA_DIR`
- `AUDITORY_LEARNING_V2_POSTGRES_DSN`
- `AUDITORY_LEARNING_V2_POSTGRES_DB`
- `AUDITORY_LEARNING_V2_POSTGRES_USER`
- `AUDITORY_LEARNING_V2_POSTGRES_PASSWORD`
- `AUDITORY_LEARNING_V2_HOST`
- `AUDITORY_LEARNING_V2_PROMPT_EXPLAIN_PATH`
- `AUDITORY_LEARNING_V2_PROMPT_SPEAK_PATH`
- `AUDITORY_LEARNING_V2_DEFAULT_MODEL_NAME`
- `AUDITORY_LEARNING_V2_DEFAULT_REASONING_EFFORT`
- `AUDITORY_LEARNING_V2_VOICEVOX_URL`
- `AUDITORY_LEARNING_V2_FALLBACK_VOICEVOX_URL`
- `AUDITORY_LEARNING_V2_FRONTEND_URL`
- `AUDITORY_LEARNING_V2_BACKEND_PORT`
- `AUDITORY_LEARNING_V2_FRONTEND_PORT`
- `AUDITORY_LEARNING_V2_DATA_DIR_HOST`
- `AUDITORY_LEARNING_V2_POSTGRES_DATA_HOST`
- `AUDITORY_LEARNING_V2_CACHE_DIR_HOST`

既定値は `v2/docker-compose.yml` と scripts 側にあります。
`AUDITORY_LEARNING_V2_PROMPT_EXPLAIN_PATH` が指すファイルの内容を、解説用の既定プロンプトとして使います。`AUDITORY_LEARNING_V2_PROMPT_SPEAK_PATH` が指すファイルの内容を、読み上げ用の既定プロンプトとして使います。どちらも相対パスはリポジトリルート基準で解決します。環境変数は `scripts/launch_v2.sh` と `docker compose` で渡します。backend は `.env` を直接読みません。frontend には backend の絶対 URL を `VITE_AUDITORY_LEARNING_V2_API_BASE_URL` で渡します。backend の CORS は `AUDITORY_LEARNING_V2_FRONTEND_URL` を優先し、必要なら `AUDITORY_LEARNING_V2_HOST` から同一 host の任意 port を許可します。`localhost` への既定値は置いていません。
`AUDITORY_LEARNING_V2_DEFAULT_MODEL_NAME` は session の既定モデル名です。既定は `gpt-5.6-luna` です。
`AUDITORY_LEARNING_V2_DEFAULT_REASONING_EFFORT` は session の既定 reasoning effort です。既定は `medium` です。
`AUDITORY_LEARNING_V2_VOICEVOX_URL` が有効ならそれを使い、無効なら `AUDITORY_LEARNING_V2_FALLBACK_VOICEVOX_URL` を使います。
開始画面では URL からの開始に加えて PDF ファイルのアップロード開始もできます。

## backend の同期と favorite

- `GET /sessions/` と `GET /sessions/{request_id}` で session 一覧と snapshot を返します。
- `GET /sessions/{request_id}/settings` と `PATCH /sessions/{request_id}/settings` で解説用プロンプト、読み上げ用プロンプト、model を更新します。
- session の詳細ブロックには 2 つのプロンプト、model に加えて、累積生成回数、処理時間、token 数、コストを表示します。
- `GET /favorites/` と `POST /favorites/{request_id}/toggle` で favorite を扱います。favorite の保存単位は session と page の組で、toggle は `page_num` を指定でき、未指定ならその session の current page を対象にします。
- `POST /sessions/{request_id}/favorite` は favorite toggle の session 版エイリアスです。
- `GET /sessions/ws` で session の snapshot と更新イベントを配信します。
- generation 開始時は `generation_started`、完了時は `generation_finished` を配信します。
- 解説とプレビューは独立して先行描画できるときは先に描画し、読み込み中はぐるぐるで待機状態を示します。
- プレビューはマウスホイールで拡大縮小できます。
- ws の状態遷移は simulator ベースの pure function テストで確認します。
- 生成キューは優先度付きで、再生対象の予約が来たら同じ task_id の既存予約を高優先度に更新します。
- `explain` はキュー投入後に worker 側でキャッシュを確認し、`regenerate` はキャッシュがあっても再生成します。

backend の Postgres は、論文の正本を `papers`、実行単位を `sessions`、処理結果を `session_results` に分けています。1 つの result は 1 つの paper と 1 つの session に属します。利用情報は `session_usage_records` に append-only で積み、session には token 数とコストの累計を持たせます。解説用プロンプトと読み上げ用プロンプトは session に別々に保持します。
