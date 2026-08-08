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

ワークスペース直下の `.env` に設定を書きます。`bash scripts/launch_v2.sh` はこの `.env` を読み込み、Docker Compose に渡します。`OPENAI_API_KEY` は解説生成に必要です。

既定値は `v2/docker-compose.yml` と `scripts/launch_v2.sh` にあります。利用者が設定できる変数は次のとおりです。

| 変数 | 説明 | 既定値 |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI API キー。解説生成に必要です。 | なし |
| `AUDITORY_LEARNING_V2_HOST` | ブラウザからアクセスするホスト名または IP アドレスです。別の PC やスマートフォンからアクセスする場合に、その端末から見えるホスト名または IP アドレスへ変更します。 | `localhost` |
| `AUDITORY_LEARNING_V2_BACKEND_PORT` | backend のホスト側ポートです。 | `8000` |
| `AUDITORY_LEARNING_V2_FRONTEND_PORT` | frontend のホスト側ポートです。 | `5174` |
| `AUDITORY_LEARNING_V2_DATA_DIR` | コンテナ内で PDF、画像、説明文、音声などを保存する場所です。 | `/workspace/_data/v2_auditory_learning` |
| `AUDITORY_LEARNING_V2_DATA_DIR_HOST` | 実行時データを保存するホスト側の場所です。 | `../_data/v2_auditory_learning` |
| `AUDITORY_LEARNING_V2_CACHE_DIR_HOST` | backend の仮想環境・uv キャッシュと frontend の依存パッケージ・npm キャッシュを保存するホスト側の場所です。 | `../_cache/v2-auditory-learning` |
| `AUDITORY_LEARNING_V2_POSTGRES_DSN` | backend が接続する Postgres の接続先です。DB 名、ユーザー、パスワードを変更する場合はこの値も合わせて変更します。 | `postgresql://v2_auditory_learning:v2_auditory_learning@postgres:5432/v2_auditory_learning` |
| `AUDITORY_LEARNING_V2_POSTGRES_DB` | Postgres のデータベース名です。 | `v2_auditory_learning` |
| `AUDITORY_LEARNING_V2_POSTGRES_USER` | Postgres のユーザー名です。 | `v2_auditory_learning` |
| `AUDITORY_LEARNING_V2_POSTGRES_PASSWORD` | Postgres のパスワードです。 | `v2_auditory_learning` |
| `AUDITORY_LEARNING_V2_POSTGRES_DATA_HOST` | Postgres のデータを保存するホスト側の場所です。 | `../_data/v2_auditory_learning/postgres` |
| `AUDITORY_LEARNING_V2_PROMPT_EXPLAIN_PATH` | 解説用の既定プロンプトのファイルです。相対パスはリポジトリルート基準で解決します。 | `prompt_explain.txt` |
| `AUDITORY_LEARNING_V2_PROMPT_SPEAK_PATH` | 読み上げ用の既定プロンプトのファイルです。相対パスはリポジトリルート基準で解決します。 | `prompt_speak.txt` |
| `AUDITORY_LEARNING_V2_DEFAULT_MODEL_NAME` | 新しい session で使う解説生成モデルです。 | `gpt-5.6-luna` |
| `AUDITORY_LEARNING_V2_DEFAULT_REASONING_EFFORT` | 新しい session で使う reasoning effort です。 | `medium` |
| `AUDITORY_LEARNING_V2_VOICEVOX_URL` | 接続先として優先する VOICEVOX URL です。空の場合は fallback を使います。 | なし |
| `AUDITORY_LEARNING_V2_FALLBACK_VOICEVOX_URL` | 優先 URL が未設定または利用できない場合の VOICEVOX URL です。 | `http://voicevox:50021` |

`AUDITORY_LEARNING_V2_FRONTEND_URL` は backend の CORS 設定用で、Compose が `AUDITORY_LEARNING_V2_HOST` と `AUDITORY_LEARNING_V2_FRONTEND_PORT` から生成します。`VITE_AUDITORY_LEARNING_V2_API_BASE_URL` は frontend が backend に接続する URL で、Compose が `AUDITORY_LEARNING_V2_HOST` と `AUDITORY_LEARNING_V2_BACKEND_PORT` から生成します。通常、この 2 つを `.env` に直接書く必要はありません。

`AUDITORY_LEARNING_V2_DATA_DIR_HOST`、`AUDITORY_LEARNING_V2_POSTGRES_DATA_HOST`、`AUDITORY_LEARNING_V2_CACHE_DIR_HOST` はホスト側の保存場所です。相対パスを指定した場合は Compose の起動位置を基準に解決されます。

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
