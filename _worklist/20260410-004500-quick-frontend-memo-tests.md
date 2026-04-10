# 作業ログ

## 目的
- quick-auditory-learning frontend のメモ同期まわりをテスト可能にする

## 対応
- メモ文字列の正規化と保存判定を helper に切り出す
- helper を unit test する

## 確認
- `cd quick-auditory-learning/frontend && npm test`
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`

## 完了
- usePaperMemo の純粋ロジックに対する unit test を追加した
