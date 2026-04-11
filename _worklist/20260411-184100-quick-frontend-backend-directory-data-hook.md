# quick-auditory-learning frontend backend directory data hook

## 目的
- `databaseReady / favorites / history / sessionSummaries` と初期読込・health poll を App.tsx から切り出す。
- UI 本体と backend directory data の取得を分ける。
- App.tsx のトップレベルを薄くする。

## 実施内容
- `src/useBackendDirectoryData.ts` を追加した。
- `src/App.tsx` から initial load / health poll / directory refresh の重複を移した。
- favorites 更新時も refresh ベースに寄せた。

## 確認
- `cd /workspace/quick-auditory-learning/frontend && npm test`
- `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
- `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`

## 結果
- すべて通過。
