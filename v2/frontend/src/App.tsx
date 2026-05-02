import {
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

import {
  ApiError,
  fetchExplanation,
  fetchPageAudio,
  fetchPageImage,
  initDocument,
  regenerateExplanation,
  type ExplainResponse,
} from "./api";
import { ObjectUrlStore } from "./objectUrlStore";
import { formatPageLabel } from "./pageState";
import { useAudioPlayer } from "./useAudioPlayer";

type LoadPageOptions = {
  requestId: string;
  page: number;
  regenerate?: boolean;
};

type ControlIconKind = "stop" | "play" | "next" | "regenerate";

function ControlIcon({ kind }: { kind: ControlIconKind }) {
  switch (kind) {
    case "stop":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="6.5" y="6.5" width="11" height="11" rx="1.4" fill="currentColor" />
        </svg>
      );
    case "play":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M8 6.8v10.4c0 .8.9 1.3 1.6.9l8.6-5.2c.7-.4.7-1.4 0-1.8L9.6 5.9c-.7-.4-1.6.1-1.6.9Z" fill="currentColor" />
        </svg>
      );
    case "next":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M8 6.8v10.4c0 .8.9 1.3 1.6.9l6.7-4c.7-.4.7-1.4 0-1.8l-6.7-4c-.7-.4-1.6.1-1.6.9Z"
            fill="currentColor"
          />
          <path d="M17.5 6.5v11" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" fill="none" />
        </svg>
      );
    case "regenerate":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6.9 8.2A7 7 0 0 1 17.3 6.1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none" />
          <path d="M17.3 6.1V9.6h-3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <path d="M17.2 15.8A7 7 0 0 1 6.8 17.9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" fill="none" />
          <path d="M6.8 17.9v-3.5h3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" fill="none" />
        </svg>
      );
  }
}

function buildErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "予期しないエラーが発生しました。";
}

function buildPaperLabel(sourceUrl: string): string | null {
  try {
    const parsedUrl = new URL(sourceUrl);
    const arxivMatch = parsedUrl.pathname.match(/\/pdf\/([^/]+)(?:\.pdf)?$/);
    if (arxivMatch?.[1]) {
      return `arXiv ${arxivMatch[1]}`;
    }
    const openReviewId = parsedUrl.searchParams.get("id");
    if (openReviewId) {
      return `OpenReview ${openReviewId}`;
    }
    const lastPathPart = parsedUrl.pathname.split("/").filter(Boolean).pop();
    if (lastPathPart) {
      return lastPathPart;
    }
  } catch {
    return null;
  }
  return null;
}

export default function App() {
  const [draftUrl, setDraftUrl] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [requestId, setRequestId] = useState<string | null>(null);
  const [maxPage, setMaxPage] = useState(1);
  const [currentPage, setCurrentPage] = useState(1);
  const [explanation, setExplanation] = useState("");
  const deferredExplanation = useDeferredValue(explanation);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [autoAdvance, setAutoAdvance] = useState(false);
  const [jumpPageValue, setJumpPageValue] = useState("1");
  const [isInitializing, setIsInitializing] = useState(false);
  const [isLoadingPage, setIsLoadingPage] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("");
  const [workspaceSplit, setWorkspaceSplit] = useState(0.5);
  const [isMobileWorkspace, setIsMobileWorkspace] = useState(() => (typeof window !== "undefined" ? window.innerWidth < 680 : false));
  const [mobileWorkspaceTab, setMobileWorkspaceTab] = useState<"explanation" | "preview">("explanation");

  const imageStoreRef = useRef(new ObjectUrlStore());
  const audioStoreRef = useRef(new ObjectUrlStore());
  const loadSequenceRef = useRef(0);
  const workspaceGridRef = useRef<HTMLElement | null>(null);
  const workspaceDraggingRef = useRef(false);

  const isInitialized = requestId !== null;

  const loadPage = async (options: LoadPageOptions): Promise<void> => {
    const sequence = ++loadSequenceRef.current;
    const page = options.page;
    setError(null);
    setCurrentPage(page);
    setAudioUrl(null);
    setStatusText(options.regenerate ? `ページ ${page} を再生成しています。` : `ページ ${page} を読み込んでいます。`);
    if (options.regenerate) {
      setIsRegenerating(true);
    } else {
      setIsLoadingPage(true);
    }

    try {
      const explanationPromise: Promise<ExplainResponse> = options.regenerate
        ? regenerateExplanation(options.requestId, page)
        : fetchExplanation(options.requestId, page);

      const imageCacheKey = `${options.requestId}:${page}`;
      const audioCacheKey = `${options.requestId}:${page}`;

      let nextImageUrl = imageStoreRef.current.get(imageCacheKey);
      if (!nextImageUrl) {
        const imageBlob = await fetchPageImage(options.requestId, page);
        nextImageUrl = imageStoreRef.current.set(imageCacheKey, imageBlob);
      }

      const explanationResponse = await explanationPromise;

      if (options.regenerate) {
        audioStoreRef.current.delete(audioCacheKey);
      }

      let nextAudioUrl = audioStoreRef.current.get(audioCacheKey);
      if (!nextAudioUrl) {
        const audioBlob = await fetchPageAudio(options.requestId, page);
        nextAudioUrl = audioStoreRef.current.set(audioCacheKey, audioBlob);
      }

      if (sequence !== loadSequenceRef.current) {
        return;
      }

      startTransition(() => {
        setExplanation(explanationResponse.explanation);
        setImageUrl(nextImageUrl);
        setAudioUrl(nextAudioUrl);
        setStatusText(options.regenerate ? `ページ ${page} の再生成が完了しました。` : `ページ ${page} を表示しています。`);
      });
    } catch (error_) {
      if (sequence !== loadSequenceRef.current) {
        return;
      }
      setError(buildErrorMessage(error_));
      setStatusText("読み込みに失敗しました。");
    } finally {
      if (sequence === loadSequenceRef.current) {
        setIsLoadingPage(false);
        setIsRegenerating(false);
      }
    }
  };

  const {
    audioRef,
    speakerEnabled,
    setSpeakerEnabled,
    isPlaying,
    volume,
    setVolume,
    playbackRate,
    setPlaybackRate,
  } = useAudioPlayer({
    src: audioUrl,
    onEnded: () => {
      if (!autoAdvance || !requestId || currentPage >= maxPage) {
        return;
      }
      void loadPage({ requestId, page: currentPage + 1 });
    },
  });

  useEffect(() => {
    return () => {
      imageStoreRef.current.clear();
      audioStoreRef.current.clear();
    };
  }, []);

  useEffect(() => {
    const updateWorkspaceMode = () => {
      setIsMobileWorkspace(window.innerWidth < 680);
    };

    updateWorkspaceMode();
    window.addEventListener("resize", updateWorkspaceMode);
    return () => window.removeEventListener("resize", updateWorkspaceMode);
  }, []);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      if (!workspaceDraggingRef.current) {
        return;
      }

      const grid = workspaceGridRef.current;
      if (!grid) {
        return;
      }

      const rect = grid.getBoundingClientRect();
      const dividerWidth = 12;
      const availableWidth = rect.width - dividerWidth;
      if (availableWidth <= 0) {
        return;
      }

      const nextSplit = (event.clientX - rect.left - dividerWidth / 2) / availableWidth;
      const clampedSplit = Math.min(1, Math.max(0, nextSplit));
      setWorkspaceSplit(clampedSplit <= 0.03 ? 0 : clampedSplit >= 0.97 ? 1 : clampedSplit);
    };

    const stopDragging = () => {
      workspaceDraggingRef.current = false;
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopDragging);
    window.addEventListener("pointercancel", stopDragging);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopDragging);
      window.removeEventListener("pointercancel", stopDragging);
    };
  }, []);

  const pageLabel = useMemo(() => formatPageLabel(currentPage, maxPage), [currentPage, maxPage]);
  const paperLabel = useMemo(() => buildPaperLabel(sourceUrl), [sourceUrl]);
  const isMainCollapsed = workspaceSplit <= 0;
  const isPreviewCollapsed = workspaceSplit >= 1;
  const workspaceGridColumns = isMainCollapsed
    ? "12px minmax(0, 1fr)"
    : isPreviewCollapsed
      ? "minmax(0, 1fr) 12px"
      : `minmax(0, ${workspaceSplit}fr) 12px minmax(0, ${1 - workspaceSplit}fr)`;

  useEffect(() => {
    setJumpPageValue(String(currentPage));
  }, [currentPage]);

  const canGoPrevious = isInitialized && currentPage > 1 && !isLoadingPage && !isInitializing;
  const canGoNext = isInitialized && currentPage < maxPage && !isLoadingPage && !isInitializing;
  const canRegenerate = isInitialized && !isLoadingPage && !isInitializing && !isRegenerating;
  const isBusy = isInitializing || isLoadingPage || isRegenerating;

  const handleStart = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedUrl = draftUrl.trim();
    if (!trimmedUrl.startsWith("http")) {
      setError("URL は http または https で始めてください。");
      return;
    }

    setIsInitializing(true);
    setError(null);
    setStatusText("PDF を初期化しています。");

    imageStoreRef.current.clear();
    audioStoreRef.current.clear();
    setExplanation("");
    setImageUrl(null);
    setAudioUrl(null);

    try {
      const response = await initDocument(trimmedUrl);
      setSourceUrl(trimmedUrl);
      setRequestId(response.request_id);
      setMaxPage(response.page_num);
      setCurrentPage(1);
      await loadPage({ requestId: response.request_id, page: 1 });
    } catch (error_) {
      setError(buildErrorMessage(error_));
      setStatusText("初期化に失敗しました。");
    } finally {
      setIsInitializing(false);
    }
  };

  const handleMovePage = async (page: number) => {
    if (!requestId || page < 1 || page > maxPage || page === currentPage) {
      return;
    }
    await loadPage({ requestId, page });
  };

  const handleJumpPage = async () => {
    const nextPage = Number.parseInt(jumpPageValue, 10);
    if (!Number.isFinite(nextPage)) {
      return;
    }
    await handleMovePage(nextPage);
  };

  const handleRegenerate = async () => {
    if (!requestId) {
      return;
    }
    await loadPage({ requestId, page: currentPage, regenerate: true });
  };

  const handleStopPlayback = () => {
    const audioElement = audioRef.current;
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
    }
    setSpeakerEnabled(false);
  };

  const handleDividerPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    workspaceDraggingRef.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const explanationWorkspaceCard = (
    <section className="card explanation-card">
      <div className="card-head sticky-head">
        <div>
          <p className="section-eyebrow">{paperLabel ? `${paperLabel} - ${pageLabel}` : pageLabel}</p>
        </div>
      </div>

      {isLoadingPage || isInitializing ? (
        <div className="loading-state" role="status" aria-live="polite">
          <span className="regenerate-spinner" aria-hidden="true" />
          <p>{isInitializing ? "初期化中" : "読み込み中"}</p>
        </div>
      ) : deferredExplanation ? (
        <div className="markdown-body">
          <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
            {deferredExplanation}
          </ReactMarkdown>
        </div>
      ) : (
        <div className="empty-state">初期化後にここへページ解説が表示されます。</div>
      )}
    </section>
  );

  const previewWorkspaceCard = (
    <section className="card preview-card">
      <div className="card-head sticky-head">
        <div>
          <p className="section-eyebrow">プレビュー</p>
        </div>
        <p className="page-meta">{pageLabel}</p>
      </div>

      {isLoadingPage || isInitializing ? (
        <div className="loading-state" role="status" aria-live="polite">
          <span className="regenerate-spinner" aria-hidden="true" />
          <div className="loading-copy">
            <p className="loading-label">{isInitializing ? "初期化中" : "読み込み中"}</p>
          </div>
        </div>
      ) : imageUrl ? (
        <div className="preview-stage">
          <img src={imageUrl} alt={`Page ${currentPage}`} />
        </div>
      ) : (
        <div className="empty-state">画像はまだ読み込まれていません。</div>
      )}
    </section>
  );

  return (
    <div className="app-shell">
      <audio ref={audioRef} src={audioUrl ?? undefined} preload="auto" />

      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <div className="app-frame">
        <header className="app-topbar">
          <div className="app-brand-row">
            <p className="app-title">AUDITORY LEARNING V2</p>
            <span className="app-subtitle">db: ok / ws: ok</span>
          </div>
          <div className="topbar-links" aria-label="navigation">
            <span className="topbar-link">お気に入り</span>
            <span className="topbar-link is-active">現在のセッション</span>
          </div>
        </header>

        {error ? <section className="card error-card">{error}</section> : null}

        <section className="card top-panel">
          <div className="top-panel-row top-panel-url">
            <form className="url-form" onSubmit={handleStart}>
              <span className="url-badge">PDF URL</span>
              <label className="field url-field" aria-label="PDF URL">
                <input
                  type="url"
                  value={draftUrl}
                  onChange={(event) => setDraftUrl(event.currentTarget.value)}
                  placeholder="https://arxiv.org/pdf/... または https://openreview.net/pdf?id=..."
                />
              </label>
              <button className="primary-button" type="submit" disabled={isInitializing}>
                {isInitializing ? "初期化中..." : "開始"}
              </button>
              {statusText ? <span className={`status-chip ${isBusy ? "is-busy" : "is-idle"}`}>{statusText}</span> : null}
            </form>
          </div>

          <div className="top-panel-row top-panel-controls">
              <div className="control-toolbar-left">
              <div className="control-icon-row">
                <button className="current-session-action-button" type="button" onClick={handleStopPlayback} disabled={!isInitialized} aria-label="停止">
                  <ControlIcon kind="stop" />
                </button>
                <button
                  className="current-session-action-button"
                  type="button"
                  onClick={() => setSpeakerEnabled((current) => !current)}
                  disabled={!isInitialized}
                  aria-label={speakerEnabled ? "音声を停止" : "音声を再生"}
                >
                  <ControlIcon kind="play" />
                </button>
                <button className="current-session-action-button" type="button" onClick={() => void handleMovePage(currentPage + 1)} disabled={!canGoNext} aria-label="次ページ">
                  <ControlIcon kind="next" />
                </button>
                <button
                  className={`current-session-action-button${isRegenerating ? " is-loading" : ""}`}
                  type="button"
                  onClick={() => void handleRegenerate()}
                  disabled={!canRegenerate}
                  aria-label="再生成"
                >
                  <ControlIcon kind="regenerate" />
                </button>
              </div>

              <div className="audio-controls" role="group" aria-label="音声設定">
                <label className="audio-control-label">
                  <span>音量</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={volume}
                    onChange={(event) => setVolume(Number(event.currentTarget.value))}
                  />
                  <strong className="audio-control-value">{Math.round(volume * 100)}%</strong>
                </label>
                <label className="audio-control-label">
                  <span>速度</span>
                  <input
                    type="range"
                    min={0.75}
                    max={2}
                    step={0.05}
                    value={playbackRate}
                    onChange={(event) => setPlaybackRate(Number(event.currentTarget.value))}
                  />
                  <strong className="audio-control-value">{playbackRate.toFixed(2)}x</strong>
                </label>
              </div>

              <div className="control-page-block">
                <div className="control-page-nav">
                  <button className="ghost-button" type="button" onClick={() => void handleMovePage(currentPage - 1)} disabled={!canGoPrevious}>
                    ← 前
                  </button>
                  <span className="page-meta">{pageLabel}</span>
                  <button className="ghost-button" type="button" onClick={() => void handleMovePage(currentPage + 1)} disabled={!canGoNext}>
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
                      onChange={(event) => setJumpPageValue(event.currentTarget.value)}
                      disabled={!isInitialized}
                    />
                    <button className="ghost-button jump-button" type="button" onClick={() => void handleJumpPage()} disabled={!isInitialized}>
                      移動
                    </button>
                  </label>
                  {!isMobileWorkspace ? (
                    <label className="toggle-row toggle-row-compact control-inline-toggle">
                      <input
                        type="checkbox"
                        checked={autoAdvance}
                        onChange={(event) => setAutoAdvance(event.currentTarget.checked)}
                        disabled={!isInitialized}
                      />
                      <span>自動送り</span>
                    </label>
                  ) : null}
                </div>

                {isMobileWorkspace ? (
                  <div className="mobile-tail-row">
                    <label className="toggle-row toggle-row-compact control-inline-toggle">
                      <input
                        type="checkbox"
                        checked={autoAdvance}
                        onChange={(event) => setAutoAdvance(event.currentTarget.checked)}
                        disabled={!isInitialized}
                      />
                      <span>自動送り</span>
                    </label>
                    <button className="icon-button favorite-button mobile-favorite-button" type="button" aria-label="お気に入り">
                      <span aria-hidden="true">♡</span>
                    </button>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="control-toolbar-right">
              {statusText ? <span className={`status-chip ${isBusy ? "is-busy" : "is-idle"}`}>{statusText}</span> : null}
              {!isMobileWorkspace ? (
                <button className="icon-button favorite-button" type="button" aria-label="お気に入り">
                  <span aria-hidden="true">♡</span>
                </button>
              ) : null}
            </div>
          </div>
        </section>

        <section
          className="workspace-grid"
          ref={workspaceGridRef}
          style={{ gridTemplateColumns: workspaceGridColumns }}
        >
          {isMobileWorkspace ? (
            <div className="workspace-mobile-switcher">
              <div className="workspace-mobile-tabs" role="tablist" aria-label="解説とプレビューの切り替え">
                <button
                  type="button"
                  role="tab"
                  aria-selected={mobileWorkspaceTab === "explanation"}
                  className={`workspace-mobile-tab${mobileWorkspaceTab === "explanation" ? " is-active" : ""}`}
                  onClick={() => setMobileWorkspaceTab("explanation")}
                >
                  解説
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={mobileWorkspaceTab === "preview"}
                  className={`workspace-mobile-tab${mobileWorkspaceTab === "preview" ? " is-active" : ""}`}
                  onClick={() => setMobileWorkspaceTab("preview")}
                >
                  プレビュー
                </button>
              </div>

              <div className="workspace-mobile-panel">
                {mobileWorkspaceTab === "explanation" ? explanationWorkspaceCard : previewWorkspaceCard}
              </div>
            </div>
          ) : (
            <>
              {!isMainCollapsed ? <main className="workspace-column workspace-main">{explanationWorkspaceCard}</main> : null}

              <div
                className="workspace-divider"
                role="separator"
                aria-orientation="vertical"
                aria-label="本文とプレビューの幅を調整"
                onPointerDown={handleDividerPointerDown}
              >
                <span className="workspace-divider-handle" />
              </div>

              {!isPreviewCollapsed ? <aside className="workspace-column workspace-preview">{previewWorkspaceCard}</aside> : null}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
