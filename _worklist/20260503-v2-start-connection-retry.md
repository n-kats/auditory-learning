# v2 start connection retry

## TODO
- backend 起動直後でも開始ボタンが失敗しにくいようにする。
- 既定プロンプト取得も backend 接続待ちで再試行する。

## 進捗
- [x] `initDocument` をネットワーク失敗時に再試行する。
- [x] `/prompt/default` の取得をネットワーク失敗時に再試行する。

## 確認
- frontend build

