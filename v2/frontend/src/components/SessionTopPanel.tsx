import { useState, type FormEvent } from "react";

import { ControlIcon } from "./ControlIcon";
import { getSpeakerToggleButtonClassName } from "../sessionTopPanelState";
import { buildUploadedPdfFileName } from "../utils/appText";

function formatElapsedMs(value: number): string {
  if (value < 1000) {
    return `${value}ms`;
  }
  return `${(value / 1000).toFixed(1)}s`;
}

type SessionTopPanelProps = {
  mode?: "start" | "session";
  draftUrl: string;
  sourceUrl: string;
  draftExplainPromptText?: string;
  draftSpeakPromptText?: string;
  speechText?: string;
  draftModelName?: string;
  draftReasoningEffort?: string;
  isBusy: boolean;
  isInitializing: boolean;
  isInitialized: boolean;
  isMobileWorkspace: boolean;
  isRegenerating: boolean;
  isPlaying: boolean;
  isSavingSessionSettings?: boolean;
  hasUnsavedChanges?: boolean;
  totalGenerationCount?: number;
  totalGenerationElapsedMs?: number;
  totalInputTokens?: number;
  totalOutputTokens?: number;
  totalCostUsd?: number;
  autoAdvance: boolean;
  playSyncEnabled: boolean;
  canGoNext: boolean;
  canGoPrevious: boolean;
  canRegenerate: boolean;
  jumpPageValue: string;
  maxPage: number;
  pageLabel: string;
  playbackRate: number;
  isFavorited: boolean;
  speakerEnabled: boolean;
  volume: number;
  onAutoAdvanceChange: (checked: boolean) => void;
  onPlaySyncEnabledChange: (checked: boolean) => void;
  onDraftUrlChange?: (value: string) => void;
  onDraftExplainPromptTextChange?: (value: string) => void;
  onDraftSpeakPromptTextChange?: (value: string) => void;
  onDraftModelNameChange?: (value: string) => void;
  onDraftReasoningEffortChange?: (value: string) => void;
  onJumpPage: () => void;
  onJumpPageValueChange: (value: string) => void;
  onMoveNext: () => void;
  onMovePrevious: () => void;
  onPlaybackRateChange: (value: number) => void;
  onRegenerate: () => void;
  onStopPlayback: () => void;
  onSaveSessionSettings?: () => void;
  onSubmit?: (event: FormEvent<HTMLFormElement>) => void;
  onUpload?: (file: File) => void;
  onToggleFavorite: () => void;
  onToggleSpeaker: () => void;
  onVolumeChange: (value: number) => void;
};

export function SessionTopPanel(props: SessionTopPanelProps) {
  const {
    mode = "start",
    draftUrl,
    sourceUrl,
    draftExplainPromptText,
    draftSpeakPromptText,
    speechText,
    draftModelName,
    draftReasoningEffort,
    isBusy,
    isInitializing,
    isInitialized,
    isMobileWorkspace,
    isRegenerating,
    isPlaying,
    isSavingSessionSettings,
    hasUnsavedChanges = false,
    totalGenerationCount = 0,
    totalGenerationElapsedMs = 0,
    totalInputTokens = 0,
    totalOutputTokens = 0,
    totalCostUsd = 0,
    autoAdvance,
    playSyncEnabled,
    canGoNext,
    canGoPrevious,
    canRegenerate,
    jumpPageValue,
    maxPage,
    pageLabel,
    playbackRate,
    isFavorited,
    speakerEnabled,
    volume,
    onAutoAdvanceChange,
    onPlaySyncEnabledChange,
    onDraftUrlChange,
    onDraftExplainPromptTextChange,
    onDraftSpeakPromptTextChange,
    onDraftModelNameChange,
    onDraftReasoningEffortChange,
    onJumpPage,
    onJumpPageValueChange,
    onMoveNext,
    onMovePrevious,
    onPlaybackRateChange,
    onRegenerate,
    onStopPlayback,
    onSaveSessionSettings,
    onSubmit,
    onUpload,
    onToggleFavorite,
    onToggleSpeaker,
    onVolumeChange,
  } = props;
  const [selectedPdfFile, setSelectedPdfFile] = useState<File | null>(null);
  const uploadFileName = selectedPdfFile?.name ?? buildUploadedPdfFileName(sourceUrl);

  const sessionDetails =
    mode === "session" && onSaveSessionSettings ? (
      <details className="search-details session-settings-details">
        <summary>詳細</summary>
        <div className="panel search-panel session-settings-panel">
          <div className="session-stats-grid" aria-label="session statistics">
            <div className="session-stat">
              <span>生成回数</span>
              <strong>{totalGenerationCount}</strong>
            </div>
            <div className="session-stat">
              <span>処理時間</span>
              <strong>{formatElapsedMs(totalGenerationElapsedMs)}</strong>
            </div>
            <div className="session-stat">
              <span>入力 token</span>
              <strong>{totalInputTokens.toLocaleString()}</strong>
            </div>
            <div className="session-stat">
              <span>出力 token</span>
              <strong>{totalOutputTokens.toLocaleString()}</strong>
            </div>
            <div className="session-stat session-stat-wide">
              <span>コスト</span>
              <strong>US$ {totalCostUsd.toFixed(6)}</strong>
            </div>
          </div>
          <div className="session-settings-divider" />
          <div className="session-prompt-grid">
            <label className="field prompt-field">
              <span>解説用プロンプト</span>
              <textarea
                value={draftExplainPromptText ?? ""}
                onChange={(event) => onDraftExplainPromptTextChange?.(event.currentTarget.value)}
                rows={10}
                spellCheck={false}
              />
            </label>
            <label className="field prompt-field">
              <span>読み上げ用プロンプト</span>
              <textarea
                value={draftSpeakPromptText ?? ""}
                onChange={(event) => onDraftSpeakPromptTextChange?.(event.currentTarget.value)}
                rows={10}
                spellCheck={false}
              />
            </label>
          </div>
          <label className="field model-field">
            <span>モデル</span>
            <input
              type="text"
              value={draftModelName ?? ""}
              onChange={(event) => onDraftModelNameChange?.(event.currentTarget.value)}
            />
          </label>
          <label className="field model-field">
            <span>Reasoning Effort</span>
            <input
              type="text"
              value={draftReasoningEffort ?? ""}
              onChange={(event) => onDraftReasoningEffortChange?.(event.currentTarget.value)}
              placeholder="例: low, medium, high"
            />
          </label>
          <button className="primary-button session-settings-save" type="button" onClick={onSaveSessionSettings} disabled={isSavingSessionSettings || !hasUnsavedChanges}>
            {isSavingSessionSettings ? "保存中..." : "保存"}
          </button>
          <div className="session-reading-block" aria-label="読み上げ文">
            <span className="session-reading-label">読み上げ文</span>
            <div className="session-reading-text">{speechText && speechText.trim().length > 0 ? speechText : "未生成"}</div>
          </div>
        </div>
      </details>
    ) : null;

  return (
    <section className="card top-panel">
      <div className="top-panel-row top-panel-url">
        <form className="url-form" onSubmit={onSubmit}>
          <span className="url-badge">PDF URL</span>
          <label className="field url-field" aria-label="PDF URL">
            <input
              type="url"
              value={draftUrl}
              onChange={(event) => onDraftUrlChange?.(event.currentTarget.value)}
              placeholder="https://arxiv.org/pdf/... または https://openreview.net/pdf?id=..."
            />
          </label>
          <button className={`start-button${mode === "session" ? " session-submit-button" : ""}`} type="submit" disabled={isInitializing}>
            {isInitializing ? "初期化中..." : "開始"}
          </button>
        </form>
      </div>

      {mode === "session" && onUpload ? (
        <div className="top-panel-row top-panel-upload">
          <div className="upload-launch-row upload-launch-row-session">
            <input
              id="session-upload-input"
              className="upload-input"
              type="file"
              accept="application/pdf"
              onChange={(event) => setSelectedPdfFile(event.currentTarget.files?.[0] ?? null)}
            />
            <div className="upload-selected-file" aria-label="選択中のPDFファイル名">
              {uploadFileName ?? ""}
            </div>
            <label className="upload-picker-button ghost-button" htmlFor="session-upload-input">
              <ControlIcon kind="upload" />
              PDF を選ぶ
            </label>
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
          </div>
        </div>
      ) : null}

      <div className="top-panel-row top-panel-controls">
        <div className="control-toolbar-left">
          <div className="control-icon-row">
            <button className="current-session-action-button" type="button" onClick={onStopPlayback} disabled={!isInitialized} aria-label="停止">
              <ControlIcon kind="stop" />
            </button>
            <button
              className={getSpeakerToggleButtonClassName(isPlaying)}
              type="button"
              onClick={onToggleSpeaker}
              disabled={!isInitialized}
              aria-label="音声を再生"
              aria-pressed={isPlaying}
            >
              <ControlIcon kind="play" />
            </button>
            <button className="current-session-action-button" type="button" onClick={onMoveNext} disabled={!canGoNext} aria-label="次ページ">
              <ControlIcon kind="next" />
            </button>
            <button
              className={`current-session-action-button${isRegenerating ? " is-loading" : ""}`}
              type="button"
              onClick={onRegenerate}
              disabled={!canRegenerate}
              aria-label="再生成"
            >
              <ControlIcon kind="regenerate" />
            </button>
          </div>

          <div className="audio-controls" role="group" aria-label="音声設定">
            <label className="audio-control-label">
              <span>音量</span>
              <input type="range" min={0} max={3} step={0.05} value={volume} onChange={(event) => onVolumeChange(Number(event.currentTarget.value))} />
              <strong className="audio-control-value">{Math.round(volume * 100)}%</strong>
            </label>
            <label className="audio-control-label">
              <span>速度</span>
              <input
                type="range"
                min={0.5}
                max={3}
                step={0.25}
                value={playbackRate}
                onChange={(event) => onPlaybackRateChange(Number(event.currentTarget.value))}
              />
              <strong className="audio-control-value">{playbackRate.toFixed(2)}x</strong>
            </label>
          </div>

          <div className="control-page-block">
            <div className="control-page-nav">
              <button className="ghost-button" type="button" onClick={onMovePrevious} disabled={!canGoPrevious}>
                ← 前
              </button>
              <span className="page-meta">{pageLabel}</span>
              <button className="ghost-button" type="button" onClick={onMoveNext} disabled={!canGoNext}>
                次 →
              </button>
              <label className="jump-form" aria-label="ページ移動">
                <span className="jump-label">移動先</span>
                <input
                  className="jump-input"
                  type="number"
                  min={1}
                  max={maxPage}
                  value={jumpPageValue}
                  onChange={(event) => onJumpPageValueChange(event.currentTarget.value)}
                  disabled={!isInitialized}
                />
                <button className="ghost-button jump-button" type="button" onClick={onJumpPage} disabled={!isInitialized}>
                  移動
                </button>
              </label>
              {!isMobileWorkspace ? (
                <>
                  <label className="toggle-row toggle-row-compact control-inline-toggle">
                    <input
                      type="checkbox"
                      checked={autoAdvance}
                      onChange={(event) => onAutoAdvanceChange(event.currentTarget.checked)}
                      disabled={!isInitialized}
                    />
                    <span>自動送り</span>
                  </label>
                  <label className="toggle-row toggle-row-compact control-inline-toggle">
                    <input
                      type="checkbox"
                      checked={playSyncEnabled}
                      onChange={(event) => onPlaySyncEnabledChange(event.currentTarget.checked)}
                      disabled={!isInitialized}
                    />
                    <span>再生同期</span>
                  </label>
                </>
              ) : null}
            </div>

            {isMobileWorkspace ? (
              <div className="mobile-tail-row">
                <label className="toggle-row toggle-row-compact control-inline-toggle">
                  <input
                    type="checkbox"
                    checked={autoAdvance}
                    onChange={(event) => onAutoAdvanceChange(event.currentTarget.checked)}
                    disabled={!isInitialized}
                  />
                  <span>自動送り</span>
                </label>
                <label className="toggle-row toggle-row-compact control-inline-toggle">
                  <input
                    type="checkbox"
                    checked={playSyncEnabled}
                    onChange={(event) => onPlaySyncEnabledChange(event.currentTarget.checked)}
                    disabled={!isInitialized}
                  />
                  <span>再生同期</span>
                </label>
                <button
                  className={`icon-button favorite-button mobile-favorite-button${isFavorited ? " is-active" : ""}`}
                  type="button"
                  aria-label={isFavorited ? "お気に入り解除" : "お気に入り"}
                  onClick={onToggleFavorite}
                >
                  <span aria-hidden="true">{isFavorited ? "♥" : "♡"}</span>
                </button>
              </div>
            ) : null}
          </div>
        </div>

          <div className="control-toolbar-right">
            {!isMobileWorkspace ? (
              <button
                className={`icon-button favorite-button${isFavorited ? " is-active" : ""}`}
                type="button"
                aria-label={isFavorited ? "お気に入り解除" : "お気に入り"}
                onClick={onToggleFavorite}
              >
                <span aria-hidden="true">{isFavorited ? "♥" : "♡"}</span>
              </button>
            ) : null}
        </div>
      </div>

      {sessionDetails ? <div className="top-panel-row top-panel-details">{sessionDetails}</div> : null}
    </section>
  );
}
