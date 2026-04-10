# 作業ログ

## 目的
- quick-auditory-learning frontend の検索結果状態判定をテスト可能にする

## 対応
- selected / queued / replayed / favorite / interactable の判定を helper 化する
- helper を unit test する

## 確認
- `cd quick-auditory-learning/frontend && npm test`
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`

## 完了
- 検索結果の状態判定に対する unit test を追加した
