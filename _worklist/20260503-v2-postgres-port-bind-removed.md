# v2 postgres port bind removed

## TODO
- postgres のホスト側 5432 バインドをやめる。

## 進捗
- [x] `v2/docker-compose.yml` から `ports` を外して `expose` にした。

## 確認
- compose 再起動

