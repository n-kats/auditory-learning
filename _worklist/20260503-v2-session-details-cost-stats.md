# 作業ログ

## 目的
- session の詳細ブロックを `設定変更` から `詳細` に戻す。
- prompt / model の編集行とは別に、生成回数・処理時間・token 数・コストを表示する。

## 進捗
- backend の session 集計値を websocket snapshot と session state に載せる。
- frontend の session 詳細ブロックを stats 行と編集行に分ける。
- start page の詳細見出しも `詳細` に揃える。

## 確認
- backend の compile と pytest
- frontend の build と必要なスモークテスト
