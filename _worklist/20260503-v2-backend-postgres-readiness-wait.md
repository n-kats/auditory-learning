# v2 backend postgres readiness wait

## TODO
- backend 起動時に Postgres の準備完了まで待つ。

## 進捗
- [x] startup で repository 初期化を待つ。
- [x] DNS 解決失敗や未起動時に retry する。

## 確認
- backend compileall

