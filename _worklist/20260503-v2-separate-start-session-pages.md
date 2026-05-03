# 2026-05-03 v2 separate start/session pages

## 目的
- `開始` と `続きから` を別ページに分ける。
- session 画面と開始画面を quick っぽく分離する。
- `App.tsx` を配線だけにする。

## 作業内容
- [x] start page と session page のページコンポーネントを作る。
- [x] App にページ切り替えを入れる。
- [x] topbar にページ遷移をつける。
- [x] 変更後に build を確認する。
- [x] docs と worklist を更新する。

## 完了条件
- `/` は開始・続きからのページになる。
- session 開始後は current session ページへ遷移する。
- `続きから` は専用の開始ページから開ける。
