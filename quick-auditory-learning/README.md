# quick-auditory-learning

`arXiv` 論文のメタデータを取り込み、検索結果から解説文と音声を生成して、session 単位で連続再生する実験用ツールです。

## 1. 簡単なツール説明

- Kaggle の `arxiv-dataset` など、`arXiv` メタデータの JSONL を取り込む
- Postgres で論文メタデータ、検索、セッション状態を管理する
- 検索候補から次の論文を選び、解説文と VOICEVOX 音声を再生する
- 同じ session を複数のブラウザで開いたときは、操作と進行状態を同期する

## 2. 環境準備（Docker）

必要なものは以下です。

- `docker`
- docker compose
- `bash` で実行できるシェル

起動は `docker compose` を使うので、まず次を確認してください。

```bash
docker compose version
```

このプロジェクトでは、永続データは `_data/`、キャッシュは `_cache/`、ログは `_tmp/` に置きます。

## 3. データ準備（Kaggle の arxiv-dataset）

`backend` は 1 行 1 JSON の JSONL を読み込みます。Kaggle の `arxiv-dataset` を使う場合は、`https://www.kaggle.com/datasets/Cornell-University/arxiv` を想定しています。ダウンロードしたメタデータの JSONL を `_data/quick_auditory_learning/arxiv.jsonl` に置いてください。

おすすめの手順は次のとおりです。

1. Kaggle にログインし、`arxiv-dataset` をダウンロードする
2. 展開して、メタデータの JSONL ファイルを取り出す
3. そのファイルを `_data/quick_auditory_learning/arxiv.jsonl` に置く

利用前に Kaggle の利用規約と、対象データセットのライセンスやデータカードの注意事項を確認してください。配布条件や再利用条件はデータセットごとに異なります。

別の場所に置く場合は、`.env` の `QUICK_AUDITORY_LEARNING_JSONL_PATH` を変更してください。

## 4. .env の記述

ワークスペース直下の `.env` に設定を書きます。`bash scripts/launch_quick_auditory_learning.sh` はこの `.env` を読み込みます。

```dotenv
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
QUICK_AUDITORY_LEARNING_JSONL_PATH=_data/quick_auditory_learning/arxiv.jsonl
QUICK_AUDITORY_LEARNING_HOST=localhost
QUICK_AUDITORY_LEARNING_BACKEND_PORT=8000
QUICK_AUDITORY_LEARNING_FRONTEND_PORT=5173
QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME=text-embedding-3-large
QUICK_AUDITORY_LEARNING_VOICEVOX_URL=http://voicevox:50021
```

- 必須: `OPENAI_API_KEY`。検索と解説生成を使うときに設定する
- 任意: `QUICK_AUDITORY_LEARNING_JSONL_PATH`。既定の `_data/quick_auditory_learning/arxiv.jsonl` 以外に JSONL を置くときに設定する
- 任意: `QUICK_AUDITORY_LEARNING_HOST`。別 PC やスマホからアクセスするとき、または `localhost` 以外のホスト名や IP アドレスで開きたいときに設定する
- 任意: `QUICK_AUDITORY_LEARNING_BACKEND_PORT`。別のアプリとポートがぶつかるときに設定する
- 任意: `QUICK_AUDITORY_LEARNING_FRONTEND_PORT`。別のアプリとポートがぶつかるときに設定する
- 任意: `QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME`。検索で使う埋め込みモデル名を変えるときに設定する
- 任意: `QUICK_AUDITORY_LEARNING_VOICEVOX_URL`。別の VOICEVOX エンジン URL を使うときに設定する

補足:

- `OPENAI_API_KEY` の代わりに `QUICK_AUDITORY_LEARNING_OPENAI_API_KEY` を使ってもよい
- `QUICK_AUDITORY_LEARNING_JSONL_PATH` を書かない場合は、既定の `_data/quick_auditory_learning/arxiv.jsonl` を使う
- 別 PC やスマホから開くときは、その端末から見えるホスト名か IP アドレスを `QUICK_AUDITORY_LEARNING_HOST` に入れる。`localhost` は自分の端末を指す
- 家の中や社内などのローカルネットワーク内だけで使うなら、その範囲からだけ見えるホスト名か IP アドレスを使い、外部公開は避ける
- 外部ネットワークから公開する場合は、必要なポートだけを開ける。Docker のポート公開設定、OS のファイアウォール、ルータの転送設定を確認し、過剰に公開しない

`.env` を使わずに環境変数で直接渡しても構いませんが、起動スクリプトを使う場合はワークスペース直下の `.env` を置くのが最も簡単です。

## 5. 起動・アクセス

起動します。

```bash
bash scripts/launch_quick_auditory_learning.sh
```

開発用に別のデータやキャッシュを使いたいときは `--dev` を付けると、`_dev` 末尾のディレクトリを使います。

起動すると、backend と frontend のログが流れます。`Ctrl-C` でログ追跡だけ止まり、コンテナはそのまま動きます。

起動後は、ブラウザで `http://localhost:5173` を開いてください。

停止します。

```bash
bash scripts/down_quick_auditory_learning.sh
```

## 補足

- `QUICK_AUDITORY_LEARNING_JSONL_PATH` を変えた場合、起動スクリプトがその親ディレクトリを bind mount します
- JSONL の初回 import と同期は backend 起動後に自動で走ります
- 手動で JSONL を import したい場合は次のコマンドを使えます

```bash
cd quick-auditory-learning/backend
uv run python -m quick_auditory_learning.cli import-jsonl /workspace/_data/quick_auditory_learning/arxiv.jsonl
```
