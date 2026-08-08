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

このプロジェクトでは、永続データは `_data/`、キャッシュは `_cache/`、ログは `_tmp/` に置きます。

## 3. データ準備（Kaggle の arxiv-dataset）

`backend` は 1 行 1 JSON の JSONL を読み込みます。Kaggle の `arxiv-dataset` を使う場合は、[Kaggle の arxiv データセット](https://www.kaggle.com/datasets/Cornell-University/arxiv) を想定しています。ダウンロードしたメタデータの JSONL を `_data/quick_auditory_learning/arxiv.jsonl` に置いてください。

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
QUICK_AUDITORY_LEARNING_VOICEVOX_URL=
QUICK_AUDITORY_LEARNING_FALLBACK_VOICEVOX_URL=http://voicevox:50021
```

- 必須: `OPENAI_API_KEY` または `QUICK_AUDITORY_LEARNING_OPENAI_API_KEY`。検索と解説生成を使うときに、どちらか一方を設定する。両方ある場合は `QUICK_AUDITORY_LEARNING_OPENAI_API_KEY` を優先する。（規定値: なし）
- 任意: `QUICK_AUDITORY_LEARNING_JSONL_PATH`。既定の `_data/quick_auditory_learning/arxiv.jsonl` 以外に JSONL を置くときに設定する。（規定値: `_data/quick_auditory_learning/arxiv.jsonl`）
- 任意: `QUICK_AUDITORY_LEARNING_HOST`。別 PC やスマホからアクセスするとき、または `localhost` 以外のホスト名や IP アドレスで開きたいときに設定する。（規定値: `localhost`）
- 任意: `QUICK_AUDITORY_LEARNING_BACKEND_PORT`。別のアプリとポートがぶつかるときに設定する。（規定値: `8000`）
- 任意: `QUICK_AUDITORY_LEARNING_FRONTEND_PORT`。別のアプリとポートがぶつかるときに設定する。（規定値: `5173`）
- 任意: `QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME`。検索で使う埋め込みモデル名を変えるときに設定する。（規定値: `text-embedding-3-large`）
- 任意: `QUICK_AUDITORY_LEARNING_LLM_MODEL`。解説生成で使うモデルを変えるときに設定する。（規定値: `gpt-5.6-luna`）
- 任意: `QUICK_AUDITORY_LEARNING_REASONING_EFFORT`。解説・検索補助生成の reasoning effort を設定する。（規定値: `medium`）
- 任意: `QUICK_AUDITORY_LEARNING_VOICEVOX_URL`。別の VOICEVOX エンジン URL を使うときに設定する。（規定値: なし）
- 任意: `QUICK_AUDITORY_LEARNING_FALLBACK_VOICEVOX_URL`。`QUICK_AUDITORY_LEARNING_VOICEVOX_URL` が未設定のときに使う VOICEVOX エンジン URL。（規定値: `http://voicevox:50021`）

別 PC やスマホから開くときは、その端末から見えるホスト名か IP アドレスを `QUICK_AUDITORY_LEARNING_HOST` に入れます。`localhost` は自分の端末を指します。

家の中や社内などのローカルネットワーク内だけで使うなら、その範囲からだけ見えるホスト名か IP アドレスを使い、外部公開は避けてください。外部ネットワークから公開する場合は、必要なポートだけを開け、Docker のポート公開設定、OS のファイアウォール、ルータの転送設定を確認してください。

`.env` を使わずに環境変数で直接渡しても構いませんが、起動スクリプトを使う場合はワークスペース直下の `.env` を置くのが最も簡単です。

## 4.1 LLM モデル候補

`QUICK_AUDITORY_LEARNING_LLM_MODEL` に指定できるのは、コスト計算できる completion model だけです。候補は次のとおりです。

- `gpt-5`
- `gpt-5-mini`
- `gpt-5-nano`
- `gpt-5.1`
- `gpt-5.1-mini`
- `gpt-5.1-nano`
- `gpt-5.2`
- `gpt-5.2-mini`
- `gpt-5.2-nano`
- `gpt-5.6`
- `gpt-5.6-sol`
- `gpt-5.6-terra`
- `gpt-5.6-luna`
- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.4-nano`
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-4o`
- `gpt-4o-mini`

## 5. 起動・アクセス

起動します。

```bash
bash scripts/launch_quick_auditory_learning.sh
```

起動すると、backend と frontend のログが流れます。`Ctrl-C` でログ追跡だけ止まり、コンテナはそのまま動きます。

起動後は、ブラウザで `http://localhost:5173` を開いてください。

停止します。

```bash
bash scripts/down_quick_auditory_learning.sh
```

## 補足

- `QUICK_AUDITORY_LEARNING_JSONL_PATH` を変えた場合、起動スクリプトがその親ディレクトリを bind mount します
- JSONL の初回 import と同期は backend 起動後に自動で走ります
