# リポジトリ ガイドライン

## 最優先条件
本リポジトリに関する文章および対話は、原則として日本語で記述する。

文字化け回避のため、絵文字や装飾用の特殊な Unicode 記号は使わず、通常の日本語文字と ASCII を中心に書く。

不明点がある場合は、勝手に決めずに確認する。

## プロジェクトの目的
公開 PDF の URL を入力すると、内容を AI が解説し、VOICEVOX で音声化して再生する Web アプリを作る。

バックエンドは FastAPI、フロントエンドは React + Mantine UI で構成し、Docker 起動とローカル開発の両方を支える。

対象は PDF の解説と音声再生であり、ローカルファイルのアップロードや汎用文書管理は扱わない。

## ディレクトリ構造
この章はリポジトリの地図であり、「どこに何があるか」と「どの情報をどこに置くか」を説明する。
構成変更があった場合は `docs/directory_structure.md` を先に更新し、この章も最新状態に保つ。

- `auditory_learning/`: FastAPI バックエンド本体。`server.py` がエントリポイントで、共通処理は `utils/` に集約する。
- `frontend/src/`: React + TypeScript + Mantine UI の実装本体。状態管理や API 呼び出しをここで完結させる。
- `frontend/dist/`: `npm run build` で生成される静的成果物。手動編集は禁止する。
- `quick-auditory-learning/`: arXiv JSONL ベースの別系統アプリのプロジェクトルート。
- `quick-auditory-learning/backend/src/quick_auditory_learning/`: arXiv JSONL の取り込み、検索、再生キューを扱う quick プロジェクトの backend 本体。
- `quick-auditory-learning/frontend/`: quick プロジェクト用の React + Vite UI 実装。
- `quick-auditory-learning/docker-compose.yml`: quick プロジェクトの Docker Compose 定義。
- `docs/`: 仕様、設計、進捗、画像を置く。開発判断の根拠を集約する。
- `docs/directory_structure.md`: 最新のディレクトリ構成と置き場の正のソース。
- `docs/workflows/`: 手順追加のワークフローガイドを置く。
- `docs/images/`: UI スクリーンショットなどの画像アセットを置く。
- `scripts/`: 起動や補助のスクリプトを置く。`launch.sh` は Docker 起動、`serve.sh` はローカル開発用、`launch_quick_auditory_learning.sh` と `down_quick_auditory_learning.sh` は quick プロジェクト用。
- `_data/`: PDF、画像、音声などの実行時キャッシュ。削除可能だがコミット禁止。
- `tests/`: pytest のテスト置き場。新規テストはここに追加する。
- `_local/`: ローカル専用補助。コミットしない。テンプレートや個人設定を置く。
- `_worklist/`: TODO、進捗、決定事項、未決、確認手順を置く作業ログ。作業単位で更新する。
- `docker/`: Docker 関連の定義を置く。
- `README.md`: 利用方法、制限、起動手順の入口。
- `Makefile`: lint、format、test の共通コマンド定義。

## 禁止事項・非推奨事項
この章は、本リポジトリで作業するコーディングエージェント向けの禁止事項を列挙する。
ここに書かれている禁止事項は、常に守る。

- 破壊的な操作は、明示的な指示がない限り行わない。例: `rm -rf`、`git reset --hard`、`git clean -fdx`。
- `cat > file` のような上書きはしない。ファイル更新は差分が残る手段で行う。
- 秘匿情報をコミットしない。`.env` や `.env.*`、認証情報ファイルは、指示がない限り読み込まない。
- 成果物にメタ発言を書かない。会話へのコメント、自己言及、読者誘導、スコープ宣言は入れない。
- コミット、プッシュ、PR 作成は、明示的な指示がない限り行わない。
- 仕様に書いていない fallback を実装・表示・通信に追加することは厳禁。
- `_data/` と `frontend/dist/` は生成物の扱いとし、必要がなければ触らず、コミット対象にも含めない。
- 既存の未整理な変更がある場合は、差分を確認してから扱い、安易に `git` コマンドを乱用しない。

## 基本ワークフロー
本リポジトリでは、作業の進め方を `docs/` と `_worklist/` に集約する。
`docs/` は仕様・設計・意思決定など「作業の判断基準」を置く場所であり、`_worklist/` は TODO と進捗のログを置く場所である。

1. 変更に着手する前に、必要なら `docs/` を更新して前提・仕様・判断基準を揃える。
2. 変更に着手する場合は、必ず `_worklist/` に作業ログを作る。小さい変更でも作業ログを残す。
3. 作業中は `_worklist/` を更新し続ける。TODO、決定事項、未決、確認手順、方針変更、取り下げなどを記録する。
4. 実装とドキュメントが食い違う場合は、独断でどちらを正とするか決めず、確認してから直す。
5. 変更によって `docs/` の内容が古くなる場合は、関連ドキュメントも更新して整合させる。
6. 新しい運用ルールや手順を取り込む必要があるときは、`docs/workflows/workflow_addition_guidelines.md` を参照して `AGENTS.md` を更新する。
7. 完了時は `_worklist/` を完了状態にし、追加確認が必要なら確認手順も残す。

完了条件:
- `docs/` の内容と実装や成果物の内容が一致している。
- `_worklist/` が最新の状態を反映している。
- 追加確認が必要な場合は、その確認手順が `_worklist/` に残っている。

## コーディング規約と命名
- Python は 3.12 を前提にし、4 スペースインデントと 120 文字制限を守る。
- `ruff format` によるダブルクオート統一と import 順序を守り、`mypy` 警告はゼロを目指す。
- 関数とモジュールは snake_case、クラスは PascalCase、設定系は `<Role>Config` を基本にする。
- フロントエンドは TypeScript 前提で、コンポーネントは `PascalCase.tsx`、カスタムフックは `useX.ts` にする。
- Mantine テーマ変数でスタイルを揃え、フロントエンドの lint を通す。

## テストガイドライン
- `tests/` に `test_<module>.py` を作成する。
- ファイルシステム操作は一時ディレクトリを fixture で差し替える。
- OpenAI、VOICEVOX、PDF ダウンロードはモックし、キュー処理やキャッシュ生成を検証する。
- 変更範囲に対して十分なカバレッジを意識し、必要に応じて `pytest --cov=auditory_learning` を確認する。
- フロントエンドは React Testing Library で主要インタラクションのスモークテストを追加し、手動検証手順も `_worklist/` に記す。

## 参考コマンド
以下は人間向けの参考情報であり、状況に応じて使い分ける。

```bash
make lint
make format
make test
bash scripts/launch.sh
(cd frontend && npm run dev)
(cd frontend && npm run build)
```
