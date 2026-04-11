# quick-frontend-app-session-state-hook

- 目的
  - `App.tsx` に残っていた session-scoped state を `useAppSessionState` に切り出す。
  - session state と audio playback state の二重管理を避ける。

- 実施内容
  - `useAppSessionState.ts` を追加。
  - `currentSessionId` / `currentPaper` / 検索状態 / コスト / 通知 / 履歴系の state を hook 側へ移動。
  - `currentAppSessionState` は playback 側の `audioUrls` / `audioIndex` / `audioDurationMs` を受けて組み立てる。
  - `applySessionViewState` / `applyAppSessionState` / `applyReplayToState` を hook 側へ寄せた。
  - `App.tsx` 側では audio playback state を別途同期する補助関数 `applyAudioPlaybackState` を追加。

- 確認
  - `cd /workspace/quick-auditory-learning/frontend && npx tsc --noEmit`
  - `cd /workspace/quick-auditory-learning/frontend && npm test`

- 結果
  - いずれも通過。
