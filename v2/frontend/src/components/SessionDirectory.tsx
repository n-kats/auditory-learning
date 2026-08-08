import { useMemo, useState, type FormEvent } from "react";

import type { SessionSummary } from "../api";
import { ControlIcon } from "./ControlIcon";
import { buildPaperLabel } from "../utils/appText";

type SessionDirectoryProps = {
  draftUrl: string;
  draftExplainPromptText: string;
  draftSpeakPromptText: string;
  modelName: string;
  reasoningEffort: string;
  sessions: SessionSummary[];
  currentSessionId: string | null;
  isInitializing: boolean;
  isLoadingSessions: boolean;
  sessionsError: string | null;
  onContinue: (session: SessionSummary) => void;
  onDraftUrlChange: (value: string) => void;
  onDraftExplainPromptTextChange: (value: string) => void;
  onDraftSpeakPromptTextChange: (value: string) => void;
  onModelNameChange: (value: string) => void;
  onReasoningEffortChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onUpload: (file: File) => void;
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
    draftSpeakPromptText,
    modelName,
    reasoningEffort,
    sessions,
    currentSessionId,
    isInitializing,
    isLoadingSessions,
    sessionsError,
    onContinue,
    onDraftUrlChange,
    onDraftExplainPromptTextChange,
    onDraftSpeakPromptTextChange,
    onModelNameChange,
    onReasoningEffortChange,
    onSubmit,
    onUpload,
  } = props;

  const [showAllSessions, setShowAllSessions] = useState(false);
  const [selectedPdfFile, setSelectedPdfFile] = useState<File | null>(null);
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

        <div className="start-upload-row">
          <button
            type="button"
            className="start-button upload-start-button"
            disabled={isInitializing || selectedPdfFile === null}
            onClick={() => {
              if (!selectedPdfFile) {
                return;
              }
              onUpload(selectedPdfFile);
            }}
          >
            {isInitializing ? "処理中..." : "Up&開始"}
          </button>
          <input
            id="start-upload-input"
            className="upload-input"
            type="file"
            accept="application/pdf"
            onChange={(event) => setSelectedPdfFile(event.currentTarget.files?.[0] ?? null)}
          />
          <label className="ghost-button" htmlFor="start-upload-input">
            <ControlIcon kind="upload" />
            PDF を選ぶ
          </label>
          <span className="upload-selected-name">
            {selectedPdfFile ? `選択中: ${selectedPdfFile.name}` : ""}
          </span>
        </div>

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
                  value={draftSpeakPromptText}
                  onChange={(event) => onDraftSpeakPromptTextChange(event.currentTarget.value)}
                  rows={10}
                  spellCheck={false}
                />
              </label>
            </div>
            <label className="field model-field">
              <span>モデル</span>
              <input type="text" value={modelName} onChange={(event) => onModelNameChange(event.currentTarget.value)} />
            </label>
            <label className="field model-field">
              <span>Reasoning Effort</span>
              <input type="text" value={reasoningEffort} onChange={(event) => onReasoningEffortChange(event.currentTarget.value)} placeholder="例: low, medium, high" />
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
                    <p className="directory-item-url">{buildPaperLabel(session.source_url) ?? session.source_url}</p>
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
