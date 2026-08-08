import { SessionTopPanel } from "../components/SessionTopPanel";
import { WorkspaceView } from "../components/WorkspaceView";
import type { UseDocumentSessionResult } from "../hooks/useDocumentSession";

type SessionPageProps = {
  session: UseDocumentSessionResult;
};

export function SessionPage({ session }: SessionPageProps) {
  return (
    <>
      <SessionTopPanel
        mode="session"
        draftUrl={session.draftUrl}
        sourceUrl={session.sourceUrl}
        draftExplainPromptText={session.draftExplainPromptText}
        draftSpeakPromptText={session.draftSpeakPromptText}
        speechText={session.speechText}
        draftModelName={session.draftModelName}
        draftReasoningEffort={session.draftReasoningEffort}
        isBusy={session.isBusy}
        isInitializing={session.isInitializing}
        isInitialized={session.requestId !== null}
        isMobileWorkspace={session.isMobileWorkspace}
        isRegenerating={session.isRegenerating}
        isPlaying={session.isPlaying}
        isSavingSessionSettings={session.isSavingSessionSettings}
        hasUnsavedChanges={session.hasUnsavedChanges}
        totalGenerationCount={session.totalGenerationCount}
        totalGenerationElapsedMs={session.totalGenerationElapsedMs}
        totalInputTokens={session.totalInputTokens}
        totalOutputTokens={session.totalOutputTokens}
        totalCostUsd={session.totalCostUsd}
        autoAdvance={session.autoAdvance}
        playSyncEnabled={session.playSyncEnabled}
        canGoNext={session.canGoNext}
        canGoPrevious={session.canGoPrevious}
        canRegenerate={session.canRegenerate}
        jumpPageValue={session.jumpPageValue}
        maxPage={session.maxPage}
        pageLabel={session.pageLabel}
        playbackRate={session.playbackRate}
        isFavorited={session.isFavorited}
        speakerEnabled={session.speakerEnabled}
        volume={session.volume}
        onAutoAdvanceChange={session.setAutoAdvance}
        onPlaySyncEnabledChange={session.setPlaySyncEnabled}
        onDraftUrlChange={session.setDraftUrl}
        onDraftExplainPromptTextChange={session.setDraftExplainPromptText}
        onDraftSpeakPromptTextChange={session.setDraftSpeakPromptText}
        onDraftModelNameChange={session.setDraftModelName}
        onDraftReasoningEffortChange={session.setDraftReasoningEffort}
        onJumpPage={session.jumpPage}
        onJumpPageValueChange={session.setJumpPageValue}
        onMoveNext={() => void session.movePage(session.currentPage + 1)}
        onMovePrevious={() => void session.movePage(session.currentPage - 1)}
        onPlaybackRateChange={session.setPlaybackRate}
        onRegenerate={session.regeneratePage}
        onStopPlayback={session.stopPlayback}
        onSaveSessionSettings={() => void session.saveSessionSettings()}
        onSubmit={session.startDocument}
        onUpload={(file) => {
          void session.startDocumentFromUpload(file);
        }}
        onToggleFavorite={() => void session.toggleFavorite()}
        onToggleSpeaker={() => session.setSpeakerEnabled(true)}
        onVolumeChange={session.setVolume}
      />

      <WorkspaceView
        ref={session.workspaceGridRef}
        currentPage={session.currentPage}
        deferredExplanation={session.deferredExplanation}
        imageUrl={session.imageUrl}
        isInitializing={session.isInitializing}
        isLoadingPage={session.isLoadingPage}
        isMainCollapsed={session.isMainCollapsed}
        isMobileWorkspace={session.isMobileWorkspace}
        isPreviewCollapsed={session.isPreviewCollapsed}
        mobileWorkspaceTab={session.mobileWorkspaceTab}
        paperLabel={session.paperLabel}
        pageLabel={session.pageLabel}
        previewZoom={session.previewZoom}
        previewPanX={session.previewPanX}
        previewPanY={session.previewPanY}
        workspaceGridColumns={session.workspaceGridColumns}
        onDividerPointerDown={session.onDividerPointerDown}
        onPreviewWheel={session.onPreviewWheel}
        onPreviewPointerDown={session.onPreviewPointerDown}
        onMobileWorkspaceTabChange={session.setMobileWorkspaceTab}
      />
    </>
  );
}
