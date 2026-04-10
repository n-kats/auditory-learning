# quick-auditory-learning

`arXiv` の JSONL アブスト一覧をローカルで取り込み、全文検索と埋め込みベクトル検索を組み合わせて session 単位で大量再生するための実験用プロジェクト。

## 目標
- JSONL をインポートして論文メタデータを保持する。
- Postgres で全文検索とベクトル検索を扱う。
- お気に入りと再生履歴を UI から操作する。
- 解説音声をキャッシュする。
- ブラウザは arXiv URL で session を開始し、WebSocket で論文、解説、音声、次の候補を受け取る。

## 起動
```bash
bash scripts/launch_quick_auditory_learning.sh
```

起動後は `backend` / `frontend` の `logs -f` を流し続けるので、`Ctrl-C` でログ追跡だけ止まる。`db` と `voicevox` は追わないのでログがうるさくならない。コンテナ停止は `down` を使う。

JSONL の初回確認と import は backend 起動後にバックグラウンドで走る。
backend は Postgres の準備待ちで起動を止めず、DB 初期化はバックグラウンドで進める。
埋め込みベクトルは検索時に必要になった論文だけを自動生成してキャッシュする。事前全件生成はしない。
検索で候補が拾えないときは、ランダムな論文を返して処理を止めない。
session の本線は `WebSocket /sessions/ws` で進める。ブラウザは `start` で session を始め、`next` で次の論文へ進み、`resume` で再接続する。

ログは `_tmp/quick_auditory_learning/logs/backend.log` にも出る。
`QUICK_AUDITORY_LEARNING_JSONL_PATH` を変えた場合、その親ディレクトリは自動で bind mount され、backend ではその同じ絶対パスとして読まれる。
JSONL import の進捗は百分率付きで console と `backend.log` に出る。
`/config` で `OPENAI_API_KEY` の有無と JSONL の存在確認を見られる。
検索や解説生成が失敗したときは、backend の `detail` と `backend.log` の両方を確認する。

session では、現在の論文の title/abstract から検索語を作って次候補を探す。候補が 0 件のときはランダムな論文を返して処理を止めない。解説は DB に保存し、音声は失敗しても止めずに準備を試みる。音声は chunk 単位で作り、UI では連続再生に見せて、再生が終わったら WebSocket で次を要求する。session が切れても `session_id` とイベント seq で再接続できる。

停止は次を使う。

```bash
bash scripts/down_quick_auditory_learning.sh
```

このプロジェクトでは Docker の named volume は使わず、永続データは `_data/`、キャッシュは `_cache/` をホスト側の bind mount で持つ。

## host/port を変える
`.env` に以下を設定してから起動する。

```dotenv
QUICK_AUDITORY_LEARNING_HOST=localhost
QUICK_AUDITORY_LEARNING_BACKEND_PORT=8000
QUICK_AUDITORY_LEARNING_FRONTEND_PORT=5173
QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME=text-embedding-3-large
QUICK_AUDITORY_LEARNING_JSONL_PATH=/path/to/arxiv.jsonl
```

- `QUICK_AUDITORY_LEARNING_HOST`: ブラウザから見える API のホスト名
- `QUICK_AUDITORY_LEARNING_BACKEND_PORT`: backend の公開ポート
- `QUICK_AUDITORY_LEARNING_FRONTEND_PORT`: frontend の公開ポート
- `QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME`: 検索で使う埋め込みモデル名の既定値
- `QUICK_AUDITORY_LEARNING_JSONL_PATH`: 取り込み対象の arXiv JSONL
- `QUICK_AUDITORY_LEARNING_JSONL_DIR_HOST`: `QUICK_AUDITORY_LEARNING_JSONL_PATH` の親ディレクトリをホスト側で bind mount するための値
- `QUICK_AUDITORY_LEARNING_VOICEVOX_URL`: docker compose では `http://voicevox:50021` を使う。`VOICEVOX_URL` でも読める。
- `/config`: backend の診断用エンドポイント。OpenAI 設定や JSONL の存在を返す。

## 開発時の主な環境変数
- `OPENAI_API_KEY`
- `QUICK_AUDITORY_LEARNING_DATA_DIR`
- `QUICK_AUDITORY_LEARNING_CACHE_DIR`
- `QUICK_AUDITORY_LEARNING_LOG_DIR`
- `QUICK_AUDITORY_LEARNING_POSTGRES_DSN`
- `QUICK_AUDITORY_LEARNING_JSONL_PATH`
- `QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME`
- `QUICK_AUDITORY_LEARNING_EXPLANATION_MODEL`
- `QUICK_AUDITORY_LEARNING_VOICEVOX_URL`
- `QUICK_AUDITORY_LEARNING_VOICEVOX_SPEAKER_ID`
- `QUICK_AUDITORY_LEARNING_VOICEVOX_SPEED_SCALE`
- `QUICK_AUDITORY_LEARNING_VOICEVOX_VOLUME_SCALE`
- `VITE_API_BASE_URL`

`OPENAI_API_KEY` か `QUICK_AUDITORY_LEARNING_OPENAI_API_KEY` のどちらかが backend に届いていれば検索と解説生成は使える。
両方無い場合でも backend は起動するが、検索と解説生成は利用できない。
その場合は `503` で理由を返す。
検索時に埋め込みが無い場合は、候補論文だけをその場で生成してキャッシュする。

## backend CLI
```bash
cd quick-auditory-learning/backend
uv run python -m quick_auditory_learning.cli import-jsonl /path/to/arxiv.jsonl
```

## データ配置
- 永続データ: `_data/quick_auditory_learning/`
- キャッシュ: `_cache/quick-auditory-learning/`
- ログ: `_tmp/quick_auditory_learning/logs/`
- Postgres は compose 内部通信のみで使い、ホストの `5432` には公開しない。
- `voicevox` も compose 内部通信のみで使い、ホストの `50021` には公開しない。

## 構成
- `backend/`: FastAPI バックエンド
- `frontend/`: React + Vite フロントエンド
- `docker-compose.yml`: 起動定義
