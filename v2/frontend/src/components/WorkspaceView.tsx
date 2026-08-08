import { forwardRef, useCallback, useEffect, useState, type PointerEvent as ReactPointerEvent } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

type WorkspaceViewProps = {
  currentPage: number;
  deferredExplanation: string;
  imageUrl: string | null;
  isInitializing: boolean;
  isLoadingPage: boolean;
  isMainCollapsed: boolean;
  isMobileWorkspace: boolean;
  isPreviewCollapsed: boolean;
  mobileWorkspaceTab: "explanation" | "preview";
  paperLabel: string | null;
  pageLabel: string;
  previewZoom: number;
  previewPanX: number;
  previewPanY: number;
  workspaceGridColumns: string;
  onDividerPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onPreviewWheel: (event: WheelEvent) => void;
  onPreviewPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onMobileWorkspaceTabChange: (tab: "explanation" | "preview") => void;
};

export const WorkspaceView = forwardRef<HTMLElement, WorkspaceViewProps>(function WorkspaceView(props, ref) {
  const [previewStageElement, setPreviewStageElement] = useState<HTMLDivElement | null>(null);
  const previewStageRef = useCallback((element: HTMLDivElement | null) => {
    setPreviewStageElement(element);
  }, []);
  const {
    currentPage,
    deferredExplanation,
    imageUrl,
    isInitializing,
    isLoadingPage,
    isMainCollapsed,
    isMobileWorkspace,
    isPreviewCollapsed,
    mobileWorkspaceTab,
    paperLabel,
    pageLabel,
    previewZoom,
    previewPanX,
    previewPanY,
    workspaceGridColumns,
    onDividerPointerDown,
    onPreviewWheel,
    onPreviewPointerDown,
    onMobileWorkspaceTabChange,
  } = props;

  useEffect(() => {
    if (!previewStageElement) {
      return;
    }

    const handleWheel = (event: WheelEvent) => {
      onPreviewWheel(event);
    };

    previewStageElement.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      previewStageElement.removeEventListener("wheel", handleWheel);
    };
  }, [onPreviewWheel, previewStageElement]);

  const explanationCard = (
    <section className="card explanation-card">
      <div className="card-head sticky-head">
        <div>
          <p className="section-eyebrow">{paperLabel ? `${paperLabel} - ${pageLabel}` : pageLabel}</p>
        </div>
      </div>

      {deferredExplanation ? (
        <div className="markdown-body">
          <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
            {deferredExplanation}
          </ReactMarkdown>
        </div>
      ) : isLoadingPage || isInitializing ? (
        <div className="loading-state" role="status" aria-live="polite">
          <span className="regenerate-spinner" aria-hidden="true" />
          <p>{isInitializing ? "初期化中" : "読み込み中"}</p>
        </div>
      ) : (
        <div className="empty-state">初期化後にここへページ解説が表示されます。</div>
      )}
    </section>
  );

  const previewCard = (
    <section className="card preview-card">
      <div className="card-head sticky-head">
        <div>
          <p className="section-eyebrow">プレビュー</p>
        </div>
        <p className="page-meta">{pageLabel}</p>
      </div>

      {imageUrl ? (
        <div
          ref={previewStageRef}
          className="preview-stage"
          onPointerDown={onPreviewPointerDown}
          title="スクロールで拡大縮小"
          style={{ cursor: previewZoom > 1 ? "grab" : "zoom-in" }}
        >
          <div
            className="preview-stage-zoom"
            style={{ transform: `translate(${previewPanX}px, ${previewPanY}px) scale(${previewZoom})` }}
          >
            <img src={imageUrl} alt={`Page ${currentPage}`} draggable={false} />
          </div>
        </div>
      ) : isLoadingPage || isInitializing ? (
        <div className="loading-state" role="status" aria-live="polite">
          <span className="regenerate-spinner" aria-hidden="true" />
          <div className="loading-copy">
            <p className="loading-label">{isInitializing ? "初期化中" : "読み込み中"}</p>
          </div>
        </div>
      ) : (
        <div className="empty-state">画像はまだ読み込まれていません。</div>
      )}
    </section>
  );

  return (
    <section className="workspace-grid" ref={ref} style={{ gridTemplateColumns: workspaceGridColumns }}>
      {isMobileWorkspace ? (
        <div className="workspace-mobile-switcher">
          <div className="workspace-mobile-tabs" role="tablist" aria-label="解説とプレビューの切り替え">
            <button
              type="button"
              role="tab"
              aria-selected={mobileWorkspaceTab === "explanation"}
              className={`workspace-mobile-tab${mobileWorkspaceTab === "explanation" ? " is-active" : ""}`}
              onClick={() => onMobileWorkspaceTabChange("explanation")}
            >
              解説
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mobileWorkspaceTab === "preview"}
              className={`workspace-mobile-tab${mobileWorkspaceTab === "preview" ? " is-active" : ""}`}
              onClick={() => onMobileWorkspaceTabChange("preview")}
            >
              プレビュー
            </button>
          </div>

          <div className="workspace-mobile-panel">{mobileWorkspaceTab === "explanation" ? explanationCard : previewCard}</div>
        </div>
      ) : (
        <>
          {!isMainCollapsed ? <main className="workspace-column workspace-main">{explanationCard}</main> : null}

          <div
            className="workspace-divider"
            role="separator"
            aria-orientation="vertical"
            aria-label="本文とプレビューの幅を調整"
            onPointerDown={onDividerPointerDown}
          >
            <span className="workspace-divider-handle" />
          </div>

          {!isPreviewCollapsed ? <aside className="workspace-column workspace-preview">{previewCard}</aside> : null}
        </>
      )}
    </section>
  );
});
