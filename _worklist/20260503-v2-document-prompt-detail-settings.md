# v2 document prompt detail settings

## TODO
- document ごとのプロンプトを詳細設定から編集できるようにする。
- 既定プロンプトは `AUDITORY_LEARNING_V2_PROMPT_PATH` のファイル内容を使う。
- 詳細設定で未入力なら既定プロンプトを使う。

## 進捗
- [x] backend に `prompt_text` を document 単位で保存する。
- [x] backend に既定プロンプト取得 API を追加する。
- [x] frontend の開始画面に `詳細設定` と `プロンプト` 入力欄を置く。
- [x] 新規開始時は詳細設定のプロンプトを送信する。
- [x] session 再開時は保存済みプロンプトを document から復元する。

## 確認
- backend テスト
- frontend build

