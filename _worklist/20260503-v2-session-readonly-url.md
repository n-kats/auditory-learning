# 2026-05-03 v2 session readonly url

## 目的
- current session では URL 入力と開始操作を無くす。
- URL は表示のみとする。

## 作業内容
- [x] `SessionTopPanel` を start/session で分ける。
- [x] session page で URL を表示専用にする。
- [x] build を確認する。
- [x] docs と worklist を更新する。

## 完了条件
- session page に URL input と `開始` ボタンが出ない。
- start page では従来どおり URL 入力と開始ができる。
