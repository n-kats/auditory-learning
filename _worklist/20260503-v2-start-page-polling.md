# 2026-05-03 v2 start page polling

## 目的
- start page の session 一覧を polling にする。
- 手動の一覧更新ボタンと補助文言を消す。

## 作業内容
- [x] `一覧更新` ボタンを削除する。
- [x] 補助文言 `開始URLを入力して session を開始してください。` を削除する。
- [x] session 一覧を polling で更新する。
- [x] build を確認する。

## 確認
- `cd /workspace/v2/frontend && npm run build`

## 完了条件
- start page に手動更新ボタンがない。
- start page に補助文言がない。
- session 一覧が自動更新される。
