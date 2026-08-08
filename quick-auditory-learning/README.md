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

最低限、OpenAI API キーと JSONL の場所を設定します。

```dotenv
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
QUICK_AUDITORY_LEARNING_JSONL_PATH=_data/quick_auditory_learning/arxiv.jsonl
```

利用できる環境変数は次のとおりです。

| 変数 | 説明 | 既定値 |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI API キーです。検索、解説生成、検索補助生成に必要です。 | なし |
| `QUICK_AUDITORY_LEARNING_OPENAI_API_KEY` | quick 専用の OpenAI API キーです。`OPENAI_API_KEY` より優先されます。 | なし |
| `QUICK_AUDITORY_LEARNING_JSONL_PATH` | 取り込む arXiv メタデータ JSONL の場所です。起動スクリプトが相対パスをホスト側の絶対パスに変換します。 | `_data/quick_auditory_learning/arxiv.jsonl` |
| `QUICK_AUDITORY_LEARNING_JSONL_DIR_HOST` | JSONL の親ディレクトリを Docker に読み取り専用で bind mount するホスト側の場所です。通常は起動スクリプトが JSONL の場所から決めます。 | JSONL の親ディレクトリ |
| `QUICK_AUDITORY_LEARNING_HOST` | ブラウザからアクセスするホスト名または IP アドレスです。 | `localhost` |
| `QUICK_AUDITORY_LEARNING_BACKEND_PORT` | backend のホスト側ポートです。 | `8000` |
| `QUICK_AUDITORY_LEARNING_FRONTEND_PORT` | frontend のホスト側ポートです。 | `5173` |
| `QUICK_AUDITORY_LEARNING_DATA_DIR` | コンテナ内で永続データを保存する場所です。 | `/workspace/_data/quick_auditory_learning` |
| `QUICK_AUDITORY_LEARNING_DATA_DIR_HOST` | Postgres などの永続データを保存するホスト側の場所です。 | `../_data/quick_auditory_learning` |
| `QUICK_AUDITORY_LEARNING_CACHE_DIR` | コンテナ内の再生成可能なキャッシュの場所です。 | `/workspace/_cache/quick-auditory-learning` |
| `QUICK_AUDITORY_LEARNING_CACHE_DIR_HOST` | backend の仮想環境・uv キャッシュと frontend の node_modules を保存するホスト側の場所です。 | `../_cache/quick-auditory-learning` |
| `QUICK_AUDITORY_LEARNING_LOG_DIR` | backend のログを保存するコンテナ内の場所です。 | `/workspace/_tmp/quick_auditory_learning/logs` |
| `QUICK_AUDITORY_LEARNING_LOG_DIR_HOST` | ログを保存するホスト側の場所です。 | `../_tmp/quick_auditory_learning/logs` |
| `QUICK_AUDITORY_LEARNING_POSTGRES_DSN` | backend が接続する Postgres の接続先です。 | `postgresql://quick_auditory_learning:quick_auditory_learning@db:5432/quick_auditory_learning` |
| `QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME` | 論文検索に使う embedding モデルです。 | `text-embedding-3-large` |
| `QUICK_AUDITORY_LEARNING_LLM_MODEL` | 解説と検索補助生成に使う completion model です。 | `gpt-5.6-luna` |
| `QUICK_AUDITORY_LEARNING_REASONING_EFFORT` | 解説と検索補助生成の reasoning effort です。 | `medium` |
| `QUICK_AUDITORY_LEARNING_VOICEVOX_URL` | 接続先として優先する VOICEVOX URL です。空の場合は fallback を使います。 | なし |
| `QUICK_AUDITORY_LEARNING_FALLBACK_VOICEVOX_URL` | 優先 URL が未設定または利用できない場合の VOICEVOX URL です。 | `http://voicevox:50021` |

`QUICK_AUDITORY_LEARNING_FRONTEND_URL` は backend の CORS 設定用で、Compose が host と frontend port から生成します。`VITE_API_BASE_URL` は frontend が backend に接続する URL で、Compose が host と backend port から生成します。`VITE_QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME` は frontend の検索フォームに渡す embedding model 名で、Compose が `QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME` から生成します。通常、これら 3 つを `.env` に直接書く必要はありません。

backend を Docker Compose 以外で直接起動する場合は、次の VOICEVOX 設定も使えます。`QUICK_AUDITORY_LEARNING_` 付きの名前を優先し、短い名前は別名です。

| 変数 | 説明 | 既定値 |
| --- | --- | --- |
| `QUICK_AUDITORY_LEARNING_VOICEVOX_SPEAKER_ID` / `VOICEVOX_SPEAKER_ID` | VOICEVOX の fallback 話者 ID です。 | `1` |
| `QUICK_AUDITORY_LEARNING_VOICEVOX_SPEED_SCALE` / `VOICEVOX_SPEED_SCALE` | VOICEVOX の読み上げ速度です。 | `1.25` |
| `QUICK_AUDITORY_LEARNING_VOICEVOX_VOLUME_SCALE` / `VOICEVOX_VOLUME_SCALE` | VOICEVOX の音量です。 | `1.0` |

別 PC やスマホから開くときは、その端末から見えるホスト名か IP アドレスを `QUICK_AUDITORY_LEARNING_HOST` に入れます。`localhost` は自分の端末を指します。

家の中や社内などのローカルネットワーク内だけで使うなら、その範囲からだけ見えるホスト名か IP アドレスを使い、外部公開は避けてください。外部ネットワークから公開する場合は、必要なポートだけを開け、Docker のポート公開設定、OS のファイアウォール、ルータの転送設定を確認してください。

`.env` を使わずに環境変数で直接渡しても構いませんが、起動スクリプトを使う場合はワークスペース直下の `.env` を置くのが最も簡単です。

`bash scripts/launch_quick_auditory_learning.sh --dev` を使う場合、`QUICK_AUDITORY_LEARNING_DATA_DIR`、`QUICK_AUDITORY_LEARNING_CACHE_DIR`、`QUICK_AUDITORY_LEARNING_LOG_DIR`、`QUICK_AUDITORY_LEARNING_JSONL_PATH` などの未設定値は `_dev` 用の場所に切り替わります。

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
