# 20260503 v2 frontend structure split

## 目的
- `App.tsx` を小さく保つために、表示部品と純粋関数を分割する。
- quick のように、state / message / command / component の役割を分ける土台を作る。

## 作業項目
- [x] `ControlIcon` と純粋ヘルパーを `utils` に切り出す。
- [x] top panel を component に分離する。
- [x] workspace の解説・プレビュー部を component に分離する。
- [x] document session の state と副作用を hook に分離する。
- [x] 既存の挙動を壊さずにビルドを通す。
- [x] ディレクトリ構成の文書と実装を一致させる。

## 確認手順
- [x] `cd /workspace/v2/frontend && npm run build`
