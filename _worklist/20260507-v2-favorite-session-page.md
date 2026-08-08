- 目的:
  - `v2` の favorite を `session_id + page_num` 単位に修正する。
  - 一覧でも session/page を明示する。

- 方針:
  - `favorites` テーブルを `session_id` と `page_num` で管理する。
  - current session の current page に対する favorite 状態を返す。
  - frontend の favorite 一覧は session id と page num を表示する。

- 確認:
  - backend / frontend のテストと build を確認する。

- 完了:
  - `favorite_pages` を `session_id + page_num` 単位で保存するように変更した。
  - current session の current page に対する favorite 状態を返すようにした。
  - favorites 一覧で session id と favorite page を表示し、個別に解除できるようにした。
  - backend tests と frontend test / build を確認した。
