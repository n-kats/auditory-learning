import { useMemo, useState, type FormEvent } from "react";

import type { SessionSummary } from "../api";

type SessionDirectoryProps = {
  draftUrl: string;
  draftExplainPromptText: string;
  draftSpeekPromptText: string;
  modelName: string;
  sessions: SessionSummary[];
  currentSessionId: string | null;
  isInitializing: boolean;
  isLoadingSessions: boolean;
  sessionsError: string | null;
  onContinue: (session: SessionSummary) => void;
  onDraftUrlChange: (value: string) => void;
  onDraftExplainPromptTextChange: (value: string) => void;
  onDraftSpeekPromptTextChange: (value: string) => void;
  onModelNameChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
};

function formatSessionLabel(session: SessionSummary): string {
  const page = session.current_page ?? 1;
  const total = session.page_num ?? 1;
  return `p. ${page} / ${total}`;
}

export function SessionDirectory(props: SessionDirectoryProps) {
  const {
    draftUrl,
    draftExplainPromptText,
    draftSpeekPromptText,
    modelName,
    sessions,
    currentSessionId,
    isInitializing,
    isLoadingSessions,
    sessionsError,
    onContinue,
    onDraftUrlChange,
    onDraftExplainPromptTextChange,
    onDraftSpeekPromptTextChange,
    onModelNameChange,
    onSubmit,
  } = props;

  const [showAllSessions, setShowAllSessions] = useState(false);
  const visibleSessions = useMemo(
    () => (showAllSessions ? sessions : sessions.slice(0, 1)),
    [sessions, showAllSessions],
  );

  return (
    <main className="directory-shell">
      <section className="card start-card">
        <div className="card-head start-card-head">
          <div>
            <h2>はじめから</h2>
          </div>
        </div>

        <form className="start-line" onSubmit={onSubmit}>
          <button type="submit" className="start-button" disabled={isInitializing}>
            {isInitializing ? "処理中..." : "開始"}
          </button>
          <label className="field directory-url-field" aria-label="PDF URL">
            <input
              type="url"
              value={draftUrl}
              onChange={(event) => onDraftUrlChange(event.currentTarget.value)}
              placeholder="https://arxiv.org/pdf/... または https://openreview.net/pdf?id=..."
            />
          </label>
        </form>

        <details className="search-details">
          <summary>詳細</summary>
          <div className="panel search-panel">
            <div className="session-prompt-grid">
              <label className="field prompt-field">
                <span>解説用プロンプト</span>
                <textarea
                  value={draftExplainPromptText}
                  onChange={(event) => onDraftExplainPromptTextChange(event.currentTarget.value)}
                  rows={10}
                  spellCheck={false}
                />
              </label>
              <label className="field prompt-field">
                <span>読み上げ用プロンプト</span>
                <textarea
                  value={draftSpeekPromptText}
                  onChange={(event) => onDraftSpeekPromptTextChange(event.currentTarget.value)}
                  rows={10}
                  spellCheck={false}
                />
              </label>
            </div>
            <label className="field model-field">
              <span>モデル</span>
              <input type="text" value={modelName} onChange={(event) => onModelNameChange(event.currentTarget.value)} />
            </label>
          </div>
        </details>
      </section>

      <section className="card session-list-card">
        <div className="card-head session-list-head">
          <div>
            <h2>続きから</h2>
            <p className="meta">最新 {visibleSessions.length} 件 / 全 {sessions.length} 件</p>
          </div>
          <button
            type="button"
            className="session-list-toggle"
            onClick={() => setShowAllSessions((current) => !current)}
            disabled={sessions.length === 0}
          >
            {showAllSessions ? "最新 1 件に戻す" : "全部表示する"}
          </button>
        </div>

        {sessionsError ? <div className="directory-error">{sessionsError}</div> : null}

        {visibleSessions.length > 0 ? (
          <ul className="session-list">
            {visibleSessions.map((session) => (
              <li key={session.request_id} className="session-row">
                <button
                  className={`session-open-button${session.request_id === currentSessionId ? " is-current" : ""}`}
                  type="button"
                  onClick={() => onContinue(session)}
                  disabled={isLoadingSessions}
                >
                  {session.request_id === currentSessionId ? "再生中" : "再開"}
                </button>
                <div className={`session-item${session.request_id === currentSessionId ? " is-current" : ""}`}>
                  <div className="session-item-main">
                    <p className="directory-item-url">{session.source_url}</p>
                    <p className="directory-item-meta">{formatSessionLabel(session)} ・ 更新 {session.updated_at}</p>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">
            {isLoadingSessions ? "session 一覧を読み込んでいます。" : "まだ session がありません。"}
          </p>
        )}
      </section>
    </main>
  );
}
