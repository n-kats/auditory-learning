# quick-auditory-learning: next 検索先読みの開始条件を再生開始後へ変更

## 目的
- `next_paper_id` が決まった瞬間ではなく、A の再生が実際に始まってから B の検索先読みを始める。
- これにより、再生前の過剰生成を避ける。

## 変更内容
- backend に `session_playback_started` イベントと `playback_started` クライアントコマンドを追加。
- `next_paper_id` 決定時には検索先読みを開始せず、`session_playback_started` を受けた時点でのみ、現在の current paper に対応する next_paper_id の検索先読みを開始する。
- frontend の audio `onPlay` から `playback_started` を backend へ送る。
- docs に通信データ型とフローを追記する。

## テスト
- backend の helper テスト:
  - `next_paper_id` 決定時に検索先読みしない
  - 再生開始後に検索先読みする
  - `playback_started` 受信時に検索先読みを開始する
- frontend の command builder テスト:
  - `playback_started` の payload 生成

## 確認結果
- `cd /workspace/quick-auditory-learning/frontend && npm test` -> 59 passed
- `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit` -> OK
- `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q` -> 58 passed

## 確認手順
- `cd /workspace/quick-auditory-learning/frontend && npm test && npx tsc --noEmit`
- `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`
