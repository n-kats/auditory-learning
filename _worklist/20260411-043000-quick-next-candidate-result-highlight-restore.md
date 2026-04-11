# 2026-04-11 next candidate result highlight restore

## 目的
- `次に再生` で選ばれた検索結果候補を、検索結果一覧内で色付き表示に戻す。
- ただし、上部の `次へ進む` ボタン自体は強調しない。

## TODO
- frontend の検索結果候補の色付けを復元する。
- docs に候補行の色付け仕様を戻す。
- 関連テストを確認する。

## 確認
- frontend の test / tsc を実行する。

## 完了
- `次に再生` で選ばれた検索結果候補の色付き表示を復元した。
- 上部の `次へ進む` ボタンの強調は付けないままにした。
- `set_next_candidate` 送信直後の楽観的な候補表示も戻した。
- docs は候補行の色付き表示に合わせて戻した。
- frontend の `npm test` と `npx tsc --noEmit` を確認した。
