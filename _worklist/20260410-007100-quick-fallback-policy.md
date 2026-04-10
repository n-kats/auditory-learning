# 20260410 quick-auditory-learning fallback policy

## 目的
arXiv API に落ちる不自然な fallback を禁止し、検索候補の random は DB 内に限定する方針を docs と実装に反映する。

## 完了
- `paper_id` 系で arXiv API を呼ばないことを維持した。
- 検索候補が足りない場合の random fallback は DB 内の論文からランダム選択に限定した。
- `docs/spec/quick_auditory_learning.md` に方針を追記した。

## 確認
- `paper_id` 系の route / memo / audio が arXiv に飛ばないことをテストで固定する。
- 検索 fallback が DB 内 random であることをテストで固定する。

## 次
- もし他の箇所でも「別ソースに落ちる fallback」があれば同じ基準で外す。
