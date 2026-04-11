# quick-auditory-learning frontend session message flow consolidation

## 目的
- `handleSessionMessage` の残りの分岐をさらに helper に寄せる。
- `paper_ready` の memo 反映、`session_costs_updated`、`session_regenerated`、`error` をより明確に分類する。
- App.tsx のメッセージ受信処理を段階的に薄くする。

## 実施内容
- `src/sessionMessageHandlers.ts` を整理した。
- `src/App.tsx` のメッセージ処理を helper の実行計画に寄せた。
- `src/sessionMessageHandlers.test.ts` を更新した。

## 確認
- `cd /workspace/quick-auditory-learning/frontend && npm test`
- `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
- `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`

## 結果
- すべて通過。
