# v2 compose postgres healthcheck

## TODO
- backend は postgres の名前解決より前に起動しないようにする。

## 進捗
- [x] postgres に healthcheck を追加する。
- [x] backend の depends_on を service_healthy にする。

## 確認
- docker compose config

