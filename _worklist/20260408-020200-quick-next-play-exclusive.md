# 2026-04-08 quick-auditory-learning: 次に再生の排他化

## 目的
- 検索結果の「次に再生」を 1 件だけに限定する。
- 明示的にキューされていない場合でも、検索結果の先頭を次の再生候補としてハイライトする。

## 対応
- backend の session queue を 1 件上書きに変更した。
- frontend では `queued_paper_ids` が空でも検索結果の先頭候補を次の再生として色付けする。
- 既存の「次に再生」ボタンは、選択中の候補を押し直すと解除、それ以外を押すと選択先を切り替える動作にした。

## 確認
- `cd quick-auditory-learning/frontend && npx tsc --noEmit`
- `python -m py_compile quick-auditory-learning/backend/src/quick_auditory_learning/db.py quick-auditory-learning/backend/src/quick_auditory_learning/main.py`

## 補足
- 画面上のハイライトは検索結果の先頭を既定候補として使う。
- `session_queued` イベントの `queued_paper_ids` は 0 件または 1 件になる。
- 次候補の自動選択は検索順位に応じた `1/rank` 重みで行う。
- `next_paper_id` は backend が決め、frontend はそれを表示だけする。
- `next_paper_id` が決まったら backend でその論文の解説・音声生成を先読みする。
- `next_paper_id` が変わった場合も、新しい論文の先読みを追加で起動する。
- `next_paper_id` が途中で変わったら、古い先読みは停止し、同じ候補に戻ったら既存キャッシュから再開する。
- コスト確定時は `session_costs_updated` を session WS へ流し、表を随時更新する。
- `session_costs_updated` は session_events にも残すので、再接続時の復元でも追える。

## 追記
- 初回開始で音声が自動再生されない問題を避けるため、`handleStart` で自動再生フラグを立て直した。
- 再生再開が無音になりやすかったので、停止時に `currentTime` を先頭へ戻し、再開ボタンから直接 `audio.play()` を呼ぶようにした。
- `続きから` はクリック直後に `session` タブへ切り替えるようにした。
- `audio` 要素自身の `volume` と `muted` も同期し、`AudioContext` の影響を受けにくくした。
- `MediaElementAudioSource` による CORS 起因の無音化を避けるため、WebAudio 経路を削除した。
