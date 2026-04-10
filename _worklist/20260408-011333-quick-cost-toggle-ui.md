# quick-auditory-learning コスト切り替え UI

## TODO
- [x] コスト表示の単位切り替えをセグメント風の見た目に直す。
- [x] 既存のテーブル表示と干渉しないように CSS を調整する。
- [x] 変更後の表示を確認する。
- [x] 色味を他のボタンに寄せて、黒寄りの見た目を外す。
- [x] コスト表の行を固定化して、処理がない場合でも表示する。
- [x] メモ保存を WebSocket 依存から外し、GET/PUT の HTTP 経路で安定化する。

## 決定事項
- コスト表示は `再生単位` と `セッション単位` の2択を、明示的な切り替えUIとして見せる。
- 切り替えはピル型のセグメントにして、選択中だけ浮かせる。
- 色は他のボタンと同じく白基調をベースにし、選択中のみ薄い緑を使う。
- 汎用のカード内ボタン指定から `cost-tab-button` を除外して、色が上書きされないようにする。
- コスト表は `table-layout: fixed` と列幅を入れて、桁数で列位置が動かないようにする。
- コスト見出しの補助文は外して、切り替えUIだけで意味を持たせる。
- コスト表は `search / embedding / explanation / audio / keyword_generation / query_generation / prefetch` を固定で並べ、未発生のものは 0 表示にする。
- メモは `GET /papers/{paper_id}/memo` で読んで `PUT /papers/{paper_id}/memo` で保存する。

## 未決
- なし。

## 確認手順
- コスト欄の2つのボタンが、単なる横並びではなく切り替え UI に見えることを確認する。
- 選択中の単位が視覚的に分かることを確認する。
- `cd quick-auditory-learning/frontend && npx tsc --noEmit` が通ることを確認する。
