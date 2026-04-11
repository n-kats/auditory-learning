# quick-auditory-learning 次候補命名整理

## 目的
- `set_next_candidate` を、次に再生する候補を指定する通信プロトコルとして扱う。
- 旧 `queue` / `dequeue` は使わない。

## 対応
- frontend の候補クリックは常に `set_next_candidate` を送る。
- frontend の state / helper 名を `nextCandidate*` 系へ寄せる。
- backend の next candidate 関連 helper 名を揃える。
- `origin="next_candidate"` の表示を `候補指定` に寄せる。

## 確認
- frontend の unit test を更新する。
- backend の set_next_candidate / next 進行テストを更新する。
- `npm test` と `npx tsc --noEmit` を通す。
- backend の pytest を通す。

## 結果
- `set_next_candidate` で次候補を指定する流れに整理した。
- frontend の next-candidate 系 state / helper / コールバック名を整理した。
- backend の next candidate 系 helper 名を整理した。
- `origin="next_candidate"` の表示は `候補指定` に統一した。
- frontend `60 passed`、backend `59 passed` を確認した。
