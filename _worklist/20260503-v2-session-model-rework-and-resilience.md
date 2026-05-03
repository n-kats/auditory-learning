# v2 session model rework and resilience

## 目的
- セッション作成直後の WebSocket 接続失敗を減らす
- お気に入りの 404 を解消する
- document / prompt / session の関係を整理し直す
- セッションに prompt と model を紐付け、途中変更できる設計へ寄せる

## 依頼内容
- [x] WebSocket connection failed の原因を点検し、接続周辺を堅牢化する
- [x] favorites の未実装または 404 を修正する
- [ ] document は論文、session は実行単位、result は document と session の両方に紐づく構造へ整理する
- [x] prompt は session で扱えるようにし、既定値は `prompt.txt` から読む
- [x] session で prompt と model を途中更新できるようにする
- [x] session の既定モデルを `gpt-5.4-mini` にし、reasoning effort の既定を `middle` として扱う
- [x] 上部ブロックに URL 編集と「その論文を再生する」操作を追加する
- [x] 設定変更用の折りたたみをボタン列の下に置く
- [x] OpenAI 呼び出しヘルパーを `gpt_utils.py` に改名する
- [x] OpenAI 呼び出し関数名を `run_gpt` に改める

## 確認
- 実装後に frontend build
- backend compile/test

## 状態
- in_progress

## メモ
- backend は session settings API を追加済み
- frontend は session 画面で prompt/model の更新と session 再生を扱う
- document/session/result の概念分離は `papers` / `sessions` / `session_results` に分ける方針へ寄せた
- DB 操作は SQLAlchemy ではなく psycopg 直叩きの repository 層で扱う
- session の既定モデルは `gpt-5.4-mini`、既定 reasoning effort は `middle` とする
- document と session の分離に合わせて result を `session_results` として永続化する
- 利用情報は `session_usage_records` に append-only で積み、session に token 数とコストの累計を持たせる
