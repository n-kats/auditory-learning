## 作業内容
- 起点論文で検索結果が出ない原因を調査した。
- `paper_search_updated` が `session_started` / `paper_ready` より先に処理されると、状態が消える経路を確認した。
- backend の session room で、pending な検索更新を immediate events の後に流すように直した。
- frontend の session state 遷移で、同一セッションの先行検索結果を `session_started` で消さないようにした。
- 状態遷移テストと websocket 順序テストを追加した。
- `session_started` では loading を落とさず、`paper_ready` で再生可能状態に移るように修正した。
- `session_started` が `paper_ready` の後に届いても current paper を消さないようにした。
- `sessionMessageState` に、同一 session の `session_started` が `paper_ready` 後に来ても current paper を保持する回帰テストを追加した。

## 確認
- `cd /workspace/quick-auditory-learning/frontend && npm test`
- `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
- `cd /workspace/quick-auditory-learning/backend && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest ../../tests/test_quick_auditory_learning.py -q`

## 結果
- frontend tests: 51 passed
- backend tests: 51 passed
