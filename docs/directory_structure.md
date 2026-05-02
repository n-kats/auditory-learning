# Directory Structure

## バックエンド
- `auditory_learning/server.py`: FastAPI エントリーポイント。画像生成と VOICEVOX 合成を順序制御。
- `auditory_learning/utils/`: GPT プロンプト処理や音声変換の補助モジュール。共通ロジックはここへ集約。
- `_data/`: PDF、画像、音声などの実行時キャッシュ。削除可能だがコミット禁止。
- `quick-auditory-learning/`: arXiv JSONL ベースの別系統アプリのプロジェクトルート。
- `quick-auditory-learning/backend/src/quick_auditory_learning/`: arXiv JSONL 取り込みと検索、音声再生を扱う別系統の FastAPI 実装本体。
- `quick-auditory-learning/backend/docker/`: quick プロジェクト用の backend イメージ定義を置く。
- `quick-auditory-learning/frontend/`: quick プロジェクト用の React + Vite 実装本体。
- `quick-auditory-learning/frontend/docker/`: quick プロジェクト用の frontend イメージ定義を置く。
- `quick-auditory-learning/docker-compose.yml`: quick プロジェクトの起動定義。
- `v2/`: v1 と同じ PDF 解説機能を quick 風の構成で分離した新しいプロジェクトルート。
- `v2/backend/src/v2_auditory_learning/`: v2 の PDF 取得、画像変換、解説生成、音声生成を扱う FastAPI 実装本体。
- `v2/backend/docker/`: v2 backend 用の Docker イメージ定義を置く。
- `v2/frontend/`: v2 backend と分離した React + Vite 実装本体。
- `v2/frontend/docker/`: v2 frontend 用の Docker イメージ定義を置く。
- `v2/docker-compose.yml`: v2 プロジェクトの起動定義。
- `_data/quick_auditory_learning/`: quick プロジェクト専用の永続データ。JSONL インポート管理や Postgres データの置き場。
- `_cache/quick-auditory-learning/`: quick プロジェクト専用のキャッシュ。venv や音声キャッシュを置く。
- `_tmp/quick_auditory_learning/logs/`: quick プロジェクト専用の実行ログ。backend のファイルログを置く。

## フロントエンド
- `frontend/src/`: React + Mantine UI の実装。状態管理や API 呼び出しをここで完結させる。
- `frontend/dist/`: `npm run build` で生成される静的成果物。手動編集禁止。

## ツールとスクリプト
- `scripts/launch.sh`: Docker ビルドとサーバ起動のエントリ。CI 相当の再現手順。
- `scripts/serve.sh`: ローカル用の uv 同期と frontend ビルドを束ねる。
- `scripts/launch_quick_auditory_learning.sh`: quick プロジェクト用の Docker Compose 起動エントリ。
- `scripts/down_quick_auditory_learning.sh`: quick プロジェクト用の Docker Compose 停止エントリ。
- `scripts/launch_v2.sh`: v2 プロジェクト用の Docker Compose 起動エントリ。
- `scripts/down_v2.sh`: v2 プロジェクト用の Docker Compose 停止エントリ。
- `Makefile`: lint/format/test の共通コマンド。

## ドキュメント
- `docs/images/`: UI スクリーンショット等のアセット。
- `docs/spec/`: 仕様を置く。近い機能の仕様をまとめてよい。
- `docs/spec/quick_auditory_learning.md`: quick-auditory-learning の論文取得、route、メモ、音声、セッションの高レベル仕様。
- `docs/spec/quick_auditory_learning_messages.md`: quick-auditory-learning の session 種別と通信データ型の仕様。
- `docs/spec/quick_auditory_learning_flow.md`: quick-auditory-learning の通信フロー仕様。
- `docs/spec/quick_auditory_learning_implementation_notes.md`: quick-auditory-learning の実装上の注意点。
- `docs/spec/quick_auditory_learning_sync_policy.md`: quick-auditory-learning の同一 session 複数クライアント同期ポリシー。
- `docs/how_to/`: 手順を置く。`how_to_<項目名>.md` の形式で作成する。
- `docs/coding_rule/`: コーディング規約を置く。言語別の規約、命名、フォーマット、例外方針を整理する。
- `docs/web/`: 外部参照ログを置く。外部情報の参照記録を残す。
- `docs/workflows/`: ExecPlan、再開手順、サンプル更新などのワークフローガイド群。
- `_worklist/`: 作業ログ、TODO、決定事項、未決、確認手順を置く。作業単位で更新する。
