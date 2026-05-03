- 目的
  - quick 側の OpenAI Responses API 呼び出しに `store=False` を明示する。

- 変更
  - `quick_auditory_learning/session_flow.py` の `responses.create(...)` 2 箇所に `store=False` を追加した。
  - `quick_auditory_learning/main.py` の explanation generation にも `store=False` を追加した。

- 確認
  - 必要なら quick backend のテストまたは起動確認を行う。
