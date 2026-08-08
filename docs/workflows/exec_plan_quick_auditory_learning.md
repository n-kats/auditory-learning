# ExecPlan: quick-auditory-learning セッション駆動化

## 目的と意図
- 目的: arXiv の JSONL アブスト一覧を取り込み、全文検索と埋め込みベクトル検索を組み合わせて、論文解説と音声再生を大量実行できる別系統の Web アプリを立ち上げる。開始点は arXiv URL の論文にし、ブラウザと backend は WebSocket でセッションを進行させる。
- ユーザー価値: 既存の PDF URL 前提アプリとは別に、ローカル JSONL を起点にした高速な論文探索と再生キュー管理を扱えるようにする。
- 確認方法: `quick-auditory-learning/docker-compose.yml` で backend / frontend / postgres を起動し、backend のヘルスチェックとフロントエンドの表示を確認する。

## 状況と前提整理
- レポジトリの現状: 既存の `v1/` は PDF 向けであり、今回の `quick-auditory-learning/` は別ディレクトリとして追加する。
- 主要ファイル/モジュール: `quick-auditory-learning/backend/src/quick_auditory_learning/`, `quick-auditory-learning/frontend/`, `quick-auditory-learning/docker-compose.yml`, `scripts/launch_quick_auditory_learning.sh`。
- 用語定義:
  - JSONL: 1 行 1 レコードの JSON 形式のアブスト一覧。
  - ルート1: キーワード検索で候補を絞り、その後に埋め込みベクトルの類似度で評価する経路。
  - ルート2: キャッシュ済みベクトルを直接使って類似度評価する経路。
- session: 開始 URL から再生終了までの連続した 1 本の進行単位。WebSocket でイベントをやり取りし、切断しても再接続できる。
- 並列化方針: 検索・ベクトル化・解説作成・音声作成・次候補先行実行は、依存しない部分をできるだけ同時に走らせる。
- 先行実行上限: 解説作成の先行実行は、アクティブな session 数程度に抑え、何ステップも先まで走らせない。
- コスト記録: 検索、ベクトル化、解説作成、音声作成、先行実行の生成時間と価格を種類別に記録し、session 単位の集計に反映する。
- 検索語生成: 通常の単純結合検索語と LLM ベースのキーワード生成検索語を両方作り、検索結果を統合する。
- 検索方式: 通常検索、LLM キーワード列、LLM 全文検索クエリを UI で選び、各方式で約 10 件ずつ検索して統合する。
- 統合重み: 検索順位に応じて 1/rank で重み付けし、ベクトル類似度とランダム要素を加えて最終候補を決める。
- 合計処理時間: stage の足し算ではなく、開始から完了までの壁時計時間を session の合計処理時間として表示する。
- 永続データ: `_data/quick_auditory_learning/` 配下に置く Postgres のデータやインポート元の管理情報。
- キャッシュ: `_cache/quick-auditory-learning/` 配下に置く venv、音声、その他の再生成可能データ。

## 作業計画
1. `docs/directory_structure.md` と `AGENTS.md` に quick プロジェクトの配置を追加する。
2. `_worklist/` に quick プロジェクト専用の作業ログを作成する。
3. `docs/workflows/exec_plan_quick_auditory_learning.md` に、実装の前提と初期マイルストーンを記録する。
4. `quick-auditory-learning/` 配下に backend / frontend / docker-compose / 起動スクリプトの雛形を作る。
5. 最低限の backend health endpoint と frontend 表示を入れ、compose 起動で確認できる状態にする。
6. ブラウザ開始セッションを WebSocket 化し、session state / event log / trail を DB で持てるようにする。
7. 検索、ベクトル化、解説作成、音声作成、次候補先行実行の並列化と、先行実行の上限を入れる。
8. 各処理種別のコスト記録と session 集計を入れ、時間と価格を後から追えるようにする。
9. 検索語は通常検索と LLM 生成の両方を走らせて統合し、処理時間は壁時計時間で見せる。
10. 通常検索・LLM キーワード列・LLM 全文検索クエリを UI で切り替えられるようにし、rank-weighted に候補を統合する。

## 具体的な作業手順
```bash
mkdir -p quick-auditory-learning/backend/src/quick_auditory_learning
mkdir -p quick-auditory-learning/frontend/src
mkdir -p _cache/quick-auditory-learning/backend
mkdir -p _data/quick_auditory_learning

bash scripts/launch_quick_auditory_learning.sh
```

## 検証と受け入れ条件
- テスト/確認手順:
- `bash scripts/launch_quick_auditory_learning.sh` が失敗せずに起動する。
- `.env` に `QUICK_AUDITORY_LEARNING_HOST` / `QUICK_AUDITORY_LEARNING_BACKEND_PORT` / `QUICK_AUDITORY_LEARNING_FRONTEND_PORT` を置くと、起動時の公開先を変更できる。
  - backend に `GET /health` を投げて 200 を返す。
  - frontend の初期画面が表示される。
  - `npx tsc --noEmit` と `npm run build` が frontend で通る。
  - `uv run pytest /workspace/tests/test_quick_auditory_learning.py` が backend で通る。
- 期待結果:
  - backend / frontend / postgres が別サービスとして起動し、ソースコードはホスト側のマウントを使って参照される。
  - `_cache/quick-auditory-learning/backend/uv_venv` と `_data/quick_auditory_learning/` が使われる。
  - UI から WebSocket で session を開始し、解説、音声、次の論文への進行を受け取れる。

## 再実行性と復旧手順
- リトライ方法: compose を止めて、`_cache/quick-auditory-learning/` と `_data/quick_auditory_learning/` の必要な範囲だけを残して再起動する。
- 失敗時の対処: backend が起動しない場合は依存関係と環境変数を見直し、frontend は node_modules キャッシュを再作成する。

## 成果物と補足
> ここに最初の起動確認ログ、compose のサービス一覧、health check の成功例を追記する。

## インターフェースと依存関係
- 変更対象 API/クラス: `quick_auditory_learning.main:app`, `quick_auditory_learning.settings.Settings`。
- 依存ライブラリ: FastAPI, OpenAI, Postgres, JSONL インポート、WebSocket、将来的なベクトル検索ライブラリ。

## 意思決定ログ
- 判断: 既存の PDF アプリとは分けて `quick-auditory-learning/` を独立プロジェクトとして置く。
  理由: 入力データ、検索方式、データ保存方針、UI の性質が異なるため。
  日付・担当: 2026-04-04 / Codex
- 判断: 永続データは `_data/quick_auditory_learning/`、再生成可能なキャッシュは `_cache/quick-auditory-learning/` に分離する。
  理由: ユーザー指定の保存場所と、実行時キャッシュの性質を分けるため。
  日付・担当: 2026-04-04 / Codex
- 判断: session は WebSocket で進行し、現在値・イベントログ・trail を DB に分けて持つ。
  理由: 切断時の再接続、途中再開、イベント再生を扱いやすくするため。
  日付・担当: 2026-04-04 / Codex

## 結果と振り返り
- 成果: backend の JSONL インポート、embedding モデル別テーブル、全文検索、ベクトル検索、favorites、history、解説生成、VOICEVOX 音声キャッシュの雛形を入れ、純関数テストを通した。frontend も `/search`、`/favorites`、`/history/recent`、`/embedding-models`、`/explanations/{paper_id}` に接続した。voice モジュールは Python 3.13 でも import できる形にし、JSONL の起動時自動同期も入れた。埋め込みは検索時に必要な候補だけを生成してモデル別テーブルに保存し、ヒットがなければランダム候補にフォールバックする方針に整理した。解説は DB に保存し、検索結果の上位数件は解説と音声を先読みし、音声は chunk 単位で作って連続再生に見せ、最後まで再生したら現在の論文の title/abstract から検索語を作って再検索し、再生トレイル内の論文は除外しながら次の論文へ自動で進む。さらに、検索・ベクトル化・解説・音声の生成コストを種類別テーブルに記録し、session 単位の総時間と概算費用も UI に反映するようにした。
- 課題: docker 実行環境での compose 起動確認と browser からの実 API 疎通、session の WebSocket 化、セッション状態とイベントログの DB 実装が残っている。
- 次のアクション: docker compose で backend / frontend / postgres / voicevox を起動し、browser から検索、解説生成、お気に入り切り替え、prev/next を確認する。
