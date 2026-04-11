# quick-auditory-learning frontend session commands refactor

## 変更
- start / stop / next / regenerate / set_next_candidate の websocket payload 生成を `sessionCommands.ts` に切り出し
- operation payload の unit test を追加

## 確認
- `cd /workspace/quick-auditory-learning/frontend && npm test`
- `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
- `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`

## 結果
- frontend と backend の既存テストは通過

## ねらい
- 状態更新と操作送信を分け、App の操作側もテストしやすくする
