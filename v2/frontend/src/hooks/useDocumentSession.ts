import {
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type RefObject,
  type PointerEvent as ReactPointerEvent,
} from "react";

import {
  fetchSessionSettings,
  updateSessionSettings,
  notifyPlaybackStarted,
  notifyPlaybackStopped,
  toWebSocketUrl,
  type SessionSyncEvent,
} from "../api";
import {
  applyDocumentSessionSyncEvent,
  createDocumentSessionSyncState,
  type DocumentSessionSyncState,
} from "../documentSessionSync";
import { loadDocumentPage } from "../documentSessionFlow";
import {
  jumpDocumentPage,
  moveDocumentPage,
  regenerateDocumentPage,
  resumeDocumentSessionByRequestId,
  startDocumentSession,
  startDocumentSessionFromUpload,
  toggleDocumentFavorite,
} from "../documentSessionActions";
import {
  applyDocumentSessionFlowEvent,
  createDocumentSessionFlowState,
  type DocumentSessionFlowEvent,
} from "../documentSessionState";
import { ObjectUrlStore } from "../objectUrlStore";
import { formatPageLabel } from "../pageState";
import { buildPaperLabel } from "../utils/appText";
import { useAudioPlayer } from "../useAudioPlayer";
import { usePromptTemplate } from "./usePromptTemplate";
import { useWorkspaceLayout } from "./useWorkspaceLayout";

type LoadPageOptions = {
  requestId: string;
  page: number;
  regenerate?: boolean;
  fromSync?: boolean;
};

export type DocumentSessionState = {
  draftUrl: string;
  draftExplainPromptText: string;
  draftSpeakPromptText: string;
  draftModelName: string;
  draftReasoningEffort: string;
  sourceUrl: string;
  requestId: string | null;
  maxPage: number;
  currentPage: number;
  totalGenerationCount: number;
  totalGenerationElapsedMs: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCostUsd: number;
  explanation: string;
  speechText: string;
  deferredExplanation: string;
  imageUrl: string | null;
  audioUrl: string | null;
  autoAdvance: boolean;
  playSyncEnabled: boolean;
  jumpPageValue: string;
  isInitializing: boolean;
  isLoadingPage: boolean;
  isRegenerating: boolean;
  error: string | null;
  audioStatusText: string;
  audioStatusError: string | null;
  generationStatusText: string;
  isFavorited: boolean;
  isMobileWorkspace: boolean;
  mobileWorkspaceTab: "explanation" | "preview";
  paperLabel: string | null;
  pageLabel: string;
  previewZoom: number;
  canGoPrevious: boolean;
  canGoNext: boolean;
  canRegenerate: boolean;
  isBusy: boolean;
  isSavingSessionSettings: boolean;
  hasUnsavedChanges: boolean;
  isMainCollapsed: boolean;
  isPreviewCollapsed: boolean;
  workspaceGridColumns: string;
  onPreviewWheel: (event: WheelEvent) => void;
  previewPanX: number;
  previewPanY: number;
  speakerEnabled: boolean;
  isPlaying: boolean;
  volume: number;
  playbackRate: number;
};

export type DocumentSessionActions = {
  audioRef: RefObject<HTMLAudioElement | null>;
  workspaceGridRef: RefObject<HTMLElement | null>;
  setDraftUrl: (value: string) => void;
  setDraftExplainPromptText: (value: string) => void;
  setDraftSpeakPromptText: (value: string) => void;
  setDraftModelName: (value: string) => void;
  setDraftReasoningEffort: (value: string) => void;
  setAutoAdvance: (value: boolean) => void;
  setPlaySyncEnabled: (value: boolean) => void;
  setJumpPageValue: (value: string) => void;
  setMobileWorkspaceTab: (value: "explanation" | "preview") => void;
  setWorkspaceSplit: (value: number) => void;
  onPreviewPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  setSpeakerEnabled: (value: boolean | ((current: boolean) => boolean)) => void;
  setVolume: (value: number) => void;
  setPlaybackRate: (value: number) => void;
  toggleFavorite: () => Promise<void>;
  saveSessionSettings: () => Promise<void>;
  startDocument: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  startDocumentFromUpload: (file: File) => Promise<void>;
  resumeDocumentByRequestId: (requestId: string) => Promise<void>;
  movePage: (page: number) => Promise<void>;
  jumpPage: () => Promise<void>;
  regeneratePage: () => Promise<void>;
  stopPlayback: () => void;
  onDividerPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
};

export type UseDocumentSessionResult = DocumentSessionState & DocumentSessionActions;

export function useDocumentSession(): UseDocumentSessionResult {
  const [flowState, setFlowState] = useState(() => createDocumentSessionFlowState());
  const deferredExplanation = useDeferredValue(flowState.explanation);
  const [autoAdvance, setAutoAdvance] = useState(false);
  const [mobileWorkspaceTab, setMobileWorkspaceTab] = useState<"explanation" | "preview">("explanation");
  const [isSavingSessionSettings, setIsSavingSessionSettings] = useState(false);
  const [playSyncEnabled, setPlaySyncEnabled] = useState(true);
  const [savedSettings, setSavedSettings] = useState({ promptExplainText: "", promptSpeakText: "", modelName: "", reasoningEffort: "" });
  const sessionSyncRef = useRef<DocumentSessionSyncState>(createDocumentSessionSyncState());
  const {
    defaultExplainPromptText,
    defaultSpeakPromptText,
    defaultModelName,
    defaultReasoningEffort,
    draftExplainPromptText,
    draftSpeakPromptText,
    draftModelName,
    draftReasoningEffort,
    setDraftExplainPromptText,
    setDraftSpeakPromptText,
    setDraftModelName,
    setDraftReasoningEffort,
  } = usePromptTemplate();
  const {
    isMobileWorkspace,
    workspaceGridRef,
    workspaceGridColumns,
    previewZoom,
    previewPanX,
    previewPanY,
    isMainCollapsed,
    isPreviewCollapsed,
    setWorkspaceSplit,
    resetPreviewPan,
    resetPreviewZoom,
    onPreviewWheel,
    onPreviewPointerDown,
    onDividerPointerDown,
  } = useWorkspaceLayout();

  const imageStoreRef = useRef(new ObjectUrlStore());
  const audioStoreRef = useRef(new ObjectUrlStore());
  const loadSequenceRef = useRef(0);
  const wsReconnectTimeoutRef = useRef<number | null>(null);
  const wsSocketRef = useRef<WebSocket | null>(null);

  const dispatchFlowEvent = (event: DocumentSessionFlowEvent) => {
    setFlowState((current) => {
      const next = applyDocumentSessionFlowEvent(current, event);
      if (next !== current) {
        const previousSyncState = sessionSyncRef.current;
        sessionSyncRef.current = {
          requestId: next.requestId,
          currentPage: next.currentPage,
          maxPage: next.maxPage,
          isFavorited: next.isFavorited,
          promptExplainText: previousSyncState.promptExplainText,
          promptSpeakText: previousSyncState.promptSpeakText,
          modelName: previousSyncState.modelName,
          reasoningEffort: previousSyncState.reasoningEffort,
          totalGenerationCount: next.totalGenerationCount,
          totalGenerationElapsedMs: next.totalGenerationElapsedMs,
          totalInputTokens: next.totalInputTokens,
          totalOutputTokens: next.totalOutputTokens,
          totalCostUsd: next.totalCostUsd,
        };
      }
      return next;
    });
  };

  const setDraftUrl = (value: string) => {
    setFlowState((current) => (current.draftUrl === value ? current : { ...current, draftUrl: value }));
  };

  const setJumpPageValue = (value: string) => {
    setFlowState((current) => (current.jumpPageValue === value ? current : { ...current, jumpPageValue: value }));
  };

  const {
    draftUrl,
    sourceUrl,
    requestId,
    maxPage,
    currentPage,
    totalGenerationCount,
    totalGenerationElapsedMs,
    totalInputTokens,
    totalOutputTokens,
    totalCostUsd,
    explanation,
    speechText,
    imageUrl,
    audioUrl,
    jumpPageValue,
    isInitializing,
    isLoadingPage,
    isRegenerating,
    error,
    audioStatusText,
    audioStatusError,
    generationStatusText,
    isFavorited,
  } = flowState;

  const isInitialized = requestId !== null;

  const {
    audioRef,
    speakerEnabled,
    isPlaying,
    setSpeakerEnabled,
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

  const loadPage = async (options: LoadPageOptions): Promise<void> => {
    resetPreviewZoom();
    resetPreviewPan();
    if (!options.fromSync) {
      void notifyPlaybackStarted(options.requestId, options.page).catch(() => undefined);
    }
    await loadDocumentPage({
      requestId: options.requestId,
      page: options.page,
      regenerate: options.regenerate,
      imageStore: imageStoreRef.current,
      audioStore: audioStoreRef.current,
      dispatchFlowEvent,
      sequenceRef: loadSequenceRef,
    });
  };

  const documentSessionDeps = {
    imageStore: imageStoreRef.current,
    audioStore: audioStoreRef.current,
    dispatchFlowEvent,
    loadPage,
  };

  useEffect(() => {
    return () => {
      imageStoreRef.current.clear();
      audioStoreRef.current.clear();
    };
  }, []);

  useEffect(() => {
    if (!requestId) {
      setFlowState((current) => (current.isFavorited ? { ...current, isFavorited: false } : current));
      setSavedSettings({ promptExplainText: "", promptSpeakText: "", modelName: "", reasoningEffort: "" });
      return;
    }

    let canceled = false;
    void (async () => {
      try {
        const settings = await fetchSessionSettings(requestId);
        if (canceled) {
          return;
        }
        setDraftExplainPromptText(settings.prompt_explain_text);
        setDraftSpeakPromptText(settings.prompt_speak_text);
        if (settings.model_name) setDraftModelName(settings.model_name);
        if (settings.reasoning_effort) setDraftReasoningEffort(settings.reasoning_effort);
        setSavedSettings({
          promptExplainText: settings.prompt_explain_text,
          promptSpeakText: settings.prompt_speak_text,
          modelName: settings.model_name ?? "",
          reasoningEffort: settings.reasoning_effort ?? "",
        });
      } catch {
        // ignore settings fetch errors for now
      }
    })();

    let closed = false;
    let retryCount = 0;

    const connect = () => {
      if (closed) {
        return;
      }
      const socket = new WebSocket(toWebSocketUrl(`/sessions/ws?request_id=${encodeURIComponent(requestId)}`));
      wsSocketRef.current = socket;

      socket.onmessage = (event) => {
        if (closed) {
          return;
        }
        try {
          const message = JSON.parse(event.data) as SessionSyncEvent;
          if (message.type === "generation_started") {
            dispatchFlowEvent({ type: "generation_started", request_id: message.request_id, page: message.page_num });
            return;
          }
          if (message.type === "generation_finished") {
            dispatchFlowEvent({ type: "generation_finished", request_id: message.request_id, page: message.page_num });
            return;
          }
          if (message.type === "playback_started") {
            if (playSyncEnabled) {
              void loadPage({ requestId: message.request_id, page: message.page_num, fromSync: true });
            }
            return;
          }
          if (message.type === "playback_stopped") {
            if (playSyncEnabled) {
              handleStopPlayback(true);
            }
            return;
          }
          const nextState = applyDocumentSessionSyncEvent(sessionSyncRef.current, message);
          if (nextState === sessionSyncRef.current) {
            return;
          }
          sessionSyncRef.current = nextState;
          setFlowState((current) => {
            if (
              current.requestId === nextState.requestId &&
              current.currentPage === nextState.currentPage &&
              current.maxPage === nextState.maxPage &&
              current.isFavorited === nextState.isFavorited &&
              current.totalGenerationCount === nextState.totalGenerationCount &&
              current.totalGenerationElapsedMs === nextState.totalGenerationElapsedMs &&
              current.totalInputTokens === nextState.totalInputTokens &&
              current.totalOutputTokens === nextState.totalOutputTokens &&
              current.totalCostUsd === nextState.totalCostUsd
            ) {
              return current;
            }
            return {
              ...current,
              requestId: nextState.requestId,
              currentPage: nextState.currentPage,
              jumpPageValue: String(nextState.currentPage),
              maxPage: nextState.maxPage,
              isFavorited: nextState.isFavorited,
              totalGenerationCount: nextState.totalGenerationCount,
              totalGenerationElapsedMs: nextState.totalGenerationElapsedMs,
              totalInputTokens: nextState.totalInputTokens,
              totalOutputTokens: nextState.totalOutputTokens,
              totalCostUsd: nextState.totalCostUsd,
            };
          });
          const hasSettingsFields =
            "prompt_explain_text" in message ||
            "prompt_speak_text" in message ||
            "model_name" in message ||
            "reasoning_effort" in message;
          if ("prompt_explain_text" in message && typeof message.prompt_explain_text === "string") {
            setDraftExplainPromptText(message.prompt_explain_text);
          }
          if ("prompt_speak_text" in message && typeof message.prompt_speak_text === "string") {
            setDraftSpeakPromptText(message.prompt_speak_text);
          }
          if ("model_name" in message && typeof message.model_name === "string") {
            setDraftModelName(message.model_name);
          }
          if ("reasoning_effort" in message && typeof message.reasoning_effort === "string") {
            setDraftReasoningEffort(message.reasoning_effort);
          }
          if (hasSettingsFields) {
            setSavedSettings({
              promptExplainText: "prompt_explain_text" in message && typeof message.prompt_explain_text === "string" ? message.prompt_explain_text : "",
              promptSpeakText: "prompt_speak_text" in message && typeof message.prompt_speak_text === "string" ? message.prompt_speak_text : "",
              modelName: "model_name" in message && typeof message.model_name === "string" ? message.model_name : "",
              reasoningEffort: "reasoning_effort" in message && typeof message.reasoning_effort === "string" ? message.reasoning_effort : "",
            });
          }
        } catch {
          // ignore malformed ws payloads
        }
      };

      const scheduleRetry = () => {
        if (closed || retryCount >= 10) {
          return;
        }
        retryCount += 1;
        if (wsReconnectTimeoutRef.current !== null) {
          window.clearTimeout(wsReconnectTimeoutRef.current);
        }
        wsReconnectTimeoutRef.current = window.setTimeout(() => {
          wsReconnectTimeoutRef.current = null;
          connect();
        }, 1000);
      };

      socket.onerror = () => {
        socket.close();
      };
      socket.onclose = () => {
        scheduleRetry();
      };
    };

    connect();

    return () => {
      canceled = true;
      closed = true;
      if (wsSocketRef.current) {
        wsSocketRef.current.close();
        wsSocketRef.current = null;
      }
      if (wsReconnectTimeoutRef.current !== null) {
        window.clearTimeout(wsReconnectTimeoutRef.current);
        wsReconnectTimeoutRef.current = null;
      }
    };
  }, [requestId]);

  const pageLabel = useMemo(() => formatPageLabel(currentPage, maxPage), [currentPage, maxPage]);
  const paperLabel = useMemo(() => buildPaperLabel(sourceUrl), [sourceUrl]);

  useEffect(() => {
    setFlowState((current) => (current.jumpPageValue === String(currentPage) ? current : { ...current, jumpPageValue: String(currentPage) }));
  }, [currentPage]);

  const canGoPrevious = isInitialized && currentPage > 1 && !isLoadingPage && !isInitializing;
  const canGoNext = isInitialized && currentPage < maxPage && !isLoadingPage && !isInitializing;
  const canRegenerate = isInitialized && !isLoadingPage && !isInitializing && !isRegenerating;
  const isBusy = isInitializing || isLoadingPage || isRegenerating;
  const hasUnsavedChanges =
    isInitialized &&
    (draftExplainPromptText !== savedSettings.promptExplainText ||
      draftSpeakPromptText !== savedSettings.promptSpeakText ||
      draftModelName !== savedSettings.modelName ||
      draftReasoningEffort !== savedSettings.reasoningEffort);

  const handleStart = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedUrl = draftUrl.trim();
    const effectiveExplainPromptText =
      draftExplainPromptText.trim().length > 0 ? draftExplainPromptText : defaultExplainPromptText;
    const effectiveSpeakPromptText =
      draftSpeakPromptText.trim().length > 0 ? draftSpeakPromptText : defaultSpeakPromptText;
    await startDocumentSession({
      trimmedUrl,
      effectiveExplainPromptText,
      effectiveSpeakPromptText,
      effectiveModelName: draftModelName.trim() || defaultModelName,
      effectiveReasoningEffort: draftReasoningEffort.trim() || defaultReasoningEffort,
      deps: documentSessionDeps,
    });
  };

  const handleStartFromUpload = async (file: File) => {
    await startDocumentSessionFromUpload({
      file,
      effectiveExplainPromptText: draftExplainPromptText.trim().length > 0 ? draftExplainPromptText : defaultExplainPromptText,
      effectiveSpeakPromptText: draftSpeakPromptText.trim().length > 0 ? draftSpeakPromptText : defaultSpeakPromptText,
      effectiveModelName: draftModelName.trim() || defaultModelName,
      effectiveReasoningEffort: draftReasoningEffort.trim() || defaultReasoningEffort,
      deps: documentSessionDeps,
    });
  };

  const handleResumeByRequestId = async (requestId: string) => {
    await resumeDocumentSessionByRequestId({
      requestId,
      deps: documentSessionDeps,
    });
  };

  const handleMovePage = async (page: number) => {
    await moveDocumentPage({
      requestId,
      page,
      currentPage,
      maxPage,
      loadPage,
    });
  };

  const handleJumpPage = async () => {
    await jumpDocumentPage({
      jumpPageValue,
      currentPage,
      maxPage,
      requestId,
      loadPage,
    });
  };

  const handleRegenerate = async () => {
    await regenerateDocumentPage({
      requestId,
      currentPage,
      loadPage,
    });
  };

  const handleStopPlayback = (fromSync = false) => {
    const audioElement = audioRef.current;
    if (audioElement) {
      audioElement.pause();
    }
    setSpeakerEnabled(false);
    if (!fromSync && requestId) {
      void notifyPlaybackStopped(requestId).catch(() => undefined);
    }
  };

  const handleToggleFavorite = async () => {
    await toggleDocumentFavorite({
      requestId,
      pageNum: currentPage,
      dispatchFlowEvent,
    });
  };

  const handleSaveSessionSettings = async () => {
    if (!requestId) {
      return;
    }
    setIsSavingSessionSettings(true);
    try {
      const response = await updateSessionSettings(requestId, {
        prompt_explain_text: draftExplainPromptText,
        prompt_speak_text: draftSpeakPromptText,
        model_name: draftModelName,
        reasoning_effort: draftReasoningEffort,
      });
      setDraftExplainPromptText(response.prompt_explain_text);
      setDraftSpeakPromptText(response.prompt_speak_text);
      if (response.model_name) setDraftModelName(response.model_name);
      if (response.reasoning_effort) setDraftReasoningEffort(response.reasoning_effort);
      setSavedSettings({
        promptExplainText: response.prompt_explain_text,
        promptSpeakText: response.prompt_speak_text,
        modelName: response.model_name ?? "",
        reasoningEffort: response.reasoning_effort ?? "",
      });
    } finally {
      setIsSavingSessionSettings(false);
    }
  };

  return {
    audioRef,
    workspaceGridRef,
    draftUrl,
    draftExplainPromptText,
    draftSpeakPromptText,
    draftModelName,
    draftReasoningEffort,
    sourceUrl,
    requestId,
    maxPage,
    currentPage,
    totalGenerationCount,
    totalGenerationElapsedMs,
    totalInputTokens,
    totalOutputTokens,
    totalCostUsd,
    explanation,
    speechText,
    deferredExplanation,
    imageUrl,
    audioUrl,
    autoAdvance,
    playSyncEnabled,
    jumpPageValue,
    isInitializing,
    isLoadingPage,
    isRegenerating,
    error,
    audioStatusText,
    audioStatusError,
    generationStatusText,
    isFavorited,
    isMobileWorkspace,
    mobileWorkspaceTab,
    paperLabel,
    pageLabel,
    canGoPrevious,
    canGoNext,
    canRegenerate,
    isBusy,
    hasUnsavedChanges,
    isSavingSessionSettings,
    isMainCollapsed,
    isPreviewCollapsed,
    workspaceGridColumns,
    previewZoom,
    previewPanX,
    previewPanY,
    speakerEnabled,
    isPlaying,
    volume,
    playbackRate,
    setWorkspaceSplit,
    onPreviewWheel,
    onPreviewPointerDown,
    setDraftUrl,
    setDraftExplainPromptText,
    setDraftSpeakPromptText,
    setDraftModelName,
    setDraftReasoningEffort,
    setAutoAdvance,
    setPlaySyncEnabled,
    setJumpPageValue,
    setMobileWorkspaceTab,
    setSpeakerEnabled,
    setVolume,
    setPlaybackRate,
    toggleFavorite: handleToggleFavorite,
    saveSessionSettings: handleSaveSessionSettings,
    startDocument: handleStart,
    startDocumentFromUpload: handleStartFromUpload,
    resumeDocumentByRequestId: handleResumeByRequestId,
    movePage: handleMovePage,
    jumpPage: handleJumpPage,
    regeneratePage: handleRegenerate,
    stopPlayback: handleStopPlayback,
    onDividerPointerDown,
  };
}
