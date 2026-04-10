# 作業ログ

## 目的
- quick-auditory-learning frontend のメモ同期を App から切り出す

## 対応
- `usePaperMemo` hook を追加し、HTTP 読み込み / 保存 / WS 更新通知 / ステータス表示を集約
- `App.tsx` はメモの表示と入力に寄せる

## 確認
- frontend の型チェックを通した

