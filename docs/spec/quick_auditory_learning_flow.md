# quick-auditory-learning 通信フロー

## 目的
session 開始から再生、検索結果更新、次候補指定、停止、再開までの流れを、UI と backend のメッセージ往復として整理する。

## 前提
- backend は session ごとにイベントを記録し、WebSocket で配信する。
- `paper_ready` は再生開始に必要な情報を返す。
- `paper_search_updated` は検索結果だけを後から更新する。
- 次の論文の検索先読みは、再生が実際に始まってから開始する。
- 検索更新は session room に bind される前に発生した場合でも pending として保持される。
- 同じ session を複数のクライアントで開いている場合、`next` や `set_next_candidate` の結果はその session の全クライアントに配信される。

## 1. 新規開始
```text
UI                      backend
| 開始ボタン             |
| start command -------->|
|                        | session_started を記録
|                        | paper_ready を記録
|<----------------------- |
| セッション画面へ遷移    |
| 再生開始                |
| playback_started ------>|
|                        | 再生開始を記録
|                        | next_paper_id があれば次の論文の検索を先読み
|                        | 検索完了後 paper_search_updated
|<----------------------- |
| 検索結果を更新          |
```

補足:
- `session_started` が先に届き、`paper_ready` が続く。
- `paper_ready` の `search_deferred=true` のとき、検索結果はまだ表示されない。
- 検索完了後の `paper_search_updated` で検索結果一覧と次候補を更新する。
- HTTP replay のあとに websocket へ `resume` を送り、同じ session room の live 更新を受け取れる状態にする。

## 2. 続きから
### 2-1. 画面の「続きから」
`続きから` ボタンは HTTP で現在状態を組み直す。

```text
UI                          backend
| セッションID入力           |
| GET /sessions/{id} -------> |
| GET /sessions/{id}/events ->|
|<--------------------------- |
| snapshot + events を replay |
| セッション画面へ遷移        |
| WebSocket を開いて live 待受 |
| type=resume, last_seq      |
|---------------------------> |
| room 参加と差分取得         |
```

### 2-2. WebSocket の自動再接続
socket 切断後の復旧では `resume` コマンドを使う。

```text
UI                      backend
| reconnect -------------->|
| type=resume, last_seq    |
|                         | その seq 以降のイベントを返す
|<------------------------|
| 差分だけ replay          |
```

## 3. 次へ進む
```text
UI                      backend
| 再生終了                |
| next command ---------->|
|                        | session_next_requested を記録
|                        | current paper を切り替え
|                        | session_advanced を記録
|                        | paper_ready を記録
|<----------------------- |
| 次の論文の再生開始       |
| playback_started ------>|
|                        | 再生開始を記録
|                        | next_paper_id があれば次の論文の検索を先読み
|                        | 検索完了後 paper_search_updated
|<----------------------- |
| 検索結果を後から更新     |
```

補足:
- `next` は current paper の再生が終わった後に送る。
- 次候補がある場合はその候補が優先される。
- `session_next_requested` の時刻を境界に、処理待ち時間を計算する。
- `next_paper_id` が決まっても、再生開始前は検索先読みを始めない。
- `session_playback_started` を受けてから、その時点の current paper に紐づく next_paper_id の検索先読みを始める。

## 4. 次候補の指定
`set_next_candidate` は次に再生する候補を指定する操作である。
選ばれた候補は検索結果一覧の候補行で色付き表示する。
### 次候補指定
```text
UI                      backend
| 候補の次に再生を押す     |
| set_next_candidate ---->|
|                        | session_next_candidate_updated を記録
|                        | next_paper_id を更新
|<----------------------- |
| 指定した候補を次候補として反映 |
```

補足:
- UI は送信直後に候補行を楽観的に選択表示してよい。backend からの `session_next_candidate_updated` で最終同期する。

## 5. 再生成
```text
UI                      backend
| 再生成ボタン             |
| regenerate command ---->|
|                        | session_regenerated を記録
|                        | paper_ready を再度記録
|<----------------------- |
| 解説と音声を作り直す    |
|                        | 検索はバックグラウンドで再実行
|                        | paper_search_updated
```

補足:
- `regenerate` の結果は同じ session の全クライアントに共有される。
- 片方のクライアントが停止中や待機中でも、別クライアントで session が進んだ場合、再生を再開したクライアントは最新の current paper に追従する。

## 6. 停止
```text
UI                      backend
| 停止ボタン               |
| stop command ---------->|
|                        | session_stopped を記録
|<----------------------- |
| 開始画面へ戻る           |
```

補足:
- 停止後は session-scoped state を破棄する。
- frontend の表示は開始画面に戻す。

## 7. memo
```text
UI                    backend
| GET /papers/{id}/memo |
|<--------------------- |
| memo 初期値を表示      |
| PUT /papers/{id}/memo |
|---------------------> |
| memo を保存            |
|<--------------------- |
| memo/ws へ broadcast   |
|<--------------------- |
| 画面に反映             |
```

## 8. コスト更新
```text
backend
| 生成処理が完了
| record_generation_cost
| session_costs_updated を送信

UI
| 受信したら session_costs / paper_costs を更新
| pending は計算中として表示
| 確定した行は calculated として表示
```

補足:
- `session_costs_updated` は session 単位の集計更新通知。
- `paper_costs` は current paper に一致する場合のみ上書きする。

## 9. 検索結果の流れ
```text
backend
| paper_ready を先に送ることがある
| 検索は別実行で進む
| 完了後に paper_search_updated を送る

UI
| paper_ready で再生を開始
| paper_search_updated で検索結果一覧を更新
```

重要:
- `paper_ready` と `paper_search_updated` の順序は前後してよい。
- backend は session room がまだ bind されていない場合、検索更新を pending に保持して後で流す。
- 次の論文の検索先読みは、`session_playback_started` を受けてから始める。
- frontend は `session_id` と `paper_id` を見て、別 session / 別 paper の結果を流用しない。
- `前の論文から検索した他の論文` は、直前の論文で得られた `rejected_candidates` を次の論文表示へ引き継いだ UI である。paper が切り替わると current paper の検索結果とは別にこの引き継ぎ表示を更新する。

## 10. 画面の状態更新順
UI は次の順で更新する。

1. メッセージ受信
2. state 更新
3. React 再描画
4. 必要な副作用を実行

このため、`session_started` は session 確立のみを表し、`paper_ready` で再生可能状態を確定する。
