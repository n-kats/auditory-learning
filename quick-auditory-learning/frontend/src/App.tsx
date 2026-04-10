import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  apiBaseUrl,
  getHealth,
  getSessionEvents,
  listFavorites,
  getSession,
  recentHistory,
  recentSessions,
  toggleFavorite,
  resolveAudioSourceUrl,
  type Paper,
  type FavoritePaperItem,
  type SearchCandidate,
  type SearchHit,
  type SessionCosts,
  type SessionEventMessage,
  type SessionSummary,
  type SessionSnapshot,
} from "./api";
import { SearchResultList } from "./SearchResultList";
import { buildCostRows, formatAudioDurationNote, formatDurationSeconds, formatUsd } from "./costDisplay";
import { useAudioPlayback } from "./useAudioPlayback";
import { usePaperMemo } from "./usePaperMemo";
import { replaySessionEvents } from "./sessionReplay";
import { useSessionSocket } from "./useSessionSocket";

type SearchFormState = {
  sourceUrl: string;
  modelName: string;
  includeOldVectors: boolean;
  useSimpleSearch: boolean;
  useKeywordSearch: boolean;
  useFulltextSearch: boolean;
  limit: number;
  route1Weight: number;
  route2Weight: number;
};

type AppTab = "start" | "favorites" | "session";
type PendingAction = "idle" | "start" | "next" | "regenerate" | "resume";

const defaultForm: SearchFormState = {
  sourceUrl: "",
  modelName: import.meta.env.VITE_QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME?.trim() || "text-embedding-3-large",
  includeOldVectors: false,
  useSimpleSearch: true,
  useKeywordSearch: true,
  useFulltextSearch: true,
  limit: 10,
  route1Weight: 0.55,
  route2Weight: 0.45,
};

function formatCategories(categories: string[]): string {
  return categories.join(", ");
}

function formatPaperOrigin(origin: string | null): string {
  if (origin === null || origin.length === 0) {
    return "unknown";
  }
  if (origin === "db") {
    return "DB";
  }
  if (origin === "arxiv") {
    return "arXiv";
  }
  if (origin === "search") {
    return "検索";
  }
  if (origin === "queue") {
    return "キュー";
  }
  if (origin === "regenerate") {
    return "再生成";
  }
  return origin;
}

function formatSearchMode(mode: string): string {
  switch (mode) {
    case "simple":
      return "通常検索";
    case "keyword_list":
      return "キーワード列";
    case "fulltext_query":
      return "全文検索クエリ";
    case "search":
      return "検索";
    default:
      return mode;
  }
}

function uniqueSearchModes(modes: string[] | undefined): string[] {
  return [...new Set((modes ?? []).filter((mode) => mode.length > 0))];
}

function formatOptionalCostValue(value: number | null | undefined, formatter: (amount: number) => string): string {
  if (value === null || value === undefined) {
    return "計算中";
  }
  return formatter(value);
}

function buildSearchResultState(params: {
  currentPaperId: string | null;
  paperId: string;
  selectedNextPaperId: string | null;
  trailPaperIds: Set<string>;
  favoritePaperIds: Set<string>;
  canInteract: boolean;
}): {
  isSelected: boolean;
  isQueued: boolean;
  isReplayed: boolean;
  isFavorite: boolean;
  canInteract: boolean;
} {
  const isSelected = params.currentPaperId === params.paperId;
  return {
    isSelected,
    isQueued: params.selectedNextPaperId === params.paperId,
    isReplayed: params.trailPaperIds.has(params.paperId) && !isSelected,
    isFavorite: params.favoritePaperIds.has(params.paperId),
    canInteract: params.canInteract,
  };
}

type SessionActionIconKind = "stop" | "play" | "next" | "regenerate" | "favorite";

function SessionActionIcon({ kind }: { kind: SessionActionIconKind }) {
  switch (kind) {
    case "stop":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" />
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
          <path d="M6.8 8.2A7 7 0 0 1 17.3 6.1" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" fill="none" />
          <path d="M17.3 6.1V9.6h-3.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <path d="M17.2 15.8A7 7 0 0 1 6.7 17.9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" fill="none" />
          <path d="M6.7 17.9v-3.5h3.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none" />
        </svg>
      );
    case "favorite":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            d="M12 20.4 4.6 13c-1.9-1.9-1.9-4.9 0-6.8 1.9-1.9 4.9-1.9 6.8 0l.6.6.6-.6c1.9-1.9 4.9-1.9 6.8 0 1.9 1.9 1.9 4.9 0 6.8Z"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinejoin="round"
            fill="none"
          />
        </svg>
      );
  }
}

export default function App() {
  const [form, setForm] = useState({
    ...defaultForm,
    modelName: import.meta.env.VITE_QUICK_AUDITORY_LEARNING_EMBEDDING_MODEL_NAME?.trim() || "text-embedding-3-large",
  });
  const [activeTab, setActiveTab] = useState<AppTab>("start");
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentPaper, setCurrentPaper] = useState<Paper | null>(null);
  const [currentPaperSource, setCurrentPaperSource] = useState<string | null>(null);
  const [simpleSearchQuery, setSimpleSearchQuery] = useState<string>("");
  const [keywordSearchQuery, setKeywordSearchQuery] = useState<string>("");
  const [fulltextSearchQuery, setFulltextSearchQuery] = useState<string>("");
  const [searchModes, setSearchModes] = useState<string[]>([]);
  const [trailPaperIds, setTrailPaperIds] = useState<string[]>([]);
  const [queuedPaperIds, setQueuedPaperIds] = useState<string[]>([]);
  const [nextPaperId, setNextPaperId] = useState<string | null>(null);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [favorites, setFavorites] = useState<FavoritePaperItem[]>([]);
  const [history, setHistory] = useState<Array<{ from_paper_id: string | null; to_paper_id: string }>>([]);
  const [sessionSummaries, setSessionSummaries] = useState<SessionSummary[]>([]);
  const [showAllSessions, setShowAllSessions] = useState(false);
  const [rejectedCandidates, setRejectedCandidates] = useState<SearchCandidate[]>([]);
  const [fallbackUsed, setFallbackUsed] = useState(false);
  const [explanation, setExplanation] = useState<string>("");
  const [paperCosts, setPaperCosts] = useState<SessionCosts | null>(null);
  const [sessionCosts, setSessionCosts] = useState<SessionCosts | null>(null);
  const [paperTitleMap, setPaperTitleMap] = useState<Record<string, string>>({});
  const [backendNotices, setBackendNotices] = useState<string[]>([]);
  const [costTab, setCostTab] = useState<"paper" | "session">("paper");
  const [loading, setLoading] = useState(false);
  const [pendingAction, setPendingAction] = useState<PendingAction>("idle");
  const [error, setError] = useState<string | null>(null);
  const [databaseReady, setDatabaseReady] = useState(false);
  const {
    audioRef,
    shouldAutoPlayRef,
    audioUrls,
    setAudioUrls,
    audioIndex,
    setAudioIndex,
    isPlaying,
    setIsPlaying,
    audioDurationMs,
    setAudioDurationMs,
    audioVolume,
    setAudioVolume,
    audioRate,
    setAudioRate,
    stopAudio,
    pauseAudio,
    resetAudio,
  } = useAudioPlayback();
  const resumeAudioRef = useRef<() => void>(() => {});
  const pauseAudioRef = useRef<() => void>(() => {});
  const stopAudioRef = useRef<() => void>(() => {});
  const audioUrlsLengthRef = useRef(0);
  const favoriteSet = useMemo(() => new Set(favorites.map((favorite) => favorite.paper_id)), [favorites]);
  const queuedPaperSet = useMemo(() => new Set(queuedPaperIds), [queuedPaperIds]);
  const selectedNextPaperId = nextPaperId;
  const trailPaperSet = useMemo(() => new Set(trailPaperIds), [trailPaperIds]);
  const {
    paperMemo,
    setPaperMemo,
    paperMemoStatus,
    paperMemoNotice,
    paperMemoRemoteValueRef,
    paperMemoDirtyRef,
  } = usePaperMemo(currentPaper?.id ?? null, (message) => {
    setError(message);
  });
  const {
    socketRef,
    sessionIdRef,
    lastEventSeqRef,
    wsStatus,
    setMessageHandler,
    openSessionSocket,
    closeSocket,
  } = useSessionSocket({
    onError: (message) => {
      setError(message);
    },
  });
  const visibleSessionSummaries = useMemo(() => {
    const sortedSessions = [...sessionSummaries].sort(
      (left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime(),
    );
    return showAllSessions ? sortedSessions : sortedSessions.slice(0, 1);
  }, [sessionSummaries, showAllSessions]);
  const hasActiveSession = currentSessionId !== null;
  const showSessionTab = hasActiveSession || loading;
  const headerStatus = [
    `db: ${databaseReady ? "ok" : "init"}`,
    `ws: ${wsStatus === "connected" ? "ok" : wsStatus}`,
  ].join(" / ");
  const refreshHistory = async () => {
    const response = await recentHistory();
    setHistory(response.transitions);
  };

  const refreshSessions = async () => {
    const response = await recentSessions();
    setSessionSummaries(response.sessions);
  };

  useEffect(() => {
    let cancelled = false;
    const loadInitialData = async () => {
      try {
        const [favoriteResponse, historyResponse, sessionsResponse] = await Promise.all([
          listFavorites(),
          recentHistory(),
          recentSessions(),
        ]);
        if (cancelled) {
          return;
        }
        setFavorites(favoriteResponse.items);
        setHistory(historyResponse.transitions);
        setSessionSummaries(sessionsResponse.sessions);
        setError(null);
      } catch (caught: unknown) {
        if (cancelled) {
          return;
        }
        setError(caught instanceof Error ? caught.message : "failed to load initial data");
      }
    };
    if (databaseReady) {
      void loadInitialData();
    }
    return () => {
      cancelled = true;
    };
  }, [databaseReady]);

  useEffect(() => {
    let cancelled = false;
    const pollHealth = async () => {
      try {
        const response = await getHealth();
        if (cancelled) {
          return;
        }
        setDatabaseReady(response.database_ready);
      } catch (caught: unknown) {
        if (cancelled) {
          return;
        }
        setDatabaseReady(false);
        setError(caught instanceof Error ? caught.message : "health check failed");
      }
    };
    void pollHealth();
    const timer = window.setInterval(() => {
      void pollHealth();
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!databaseReady) {
      return;
    }
    void refreshSessions().catch(() => {
      // 起動直後に backend がまだ追いついていない場合は次回の polling に任せる
    });
    const timer = window.setInterval(() => {
      void refreshSessions().catch(() => {
        // 一時的な失敗は無視する
      });
    }, 5000);
    return () => window.clearInterval(timer);
  }, [databaseReady]);

  useEffect(() => {
    if (currentPaper) {
      return;
    }
    setPaperCosts(null);
    setSessionCosts(null);
    setQueuedPaperIds([]);
    setNextPaperId(null);
  }, [currentPaper?.id]);

  const applyPaperReadyEvent = async (message: SessionEventMessage) => {
    const paper = message.paper;
    if (!paper) {
      return;
    }
    const sessionId = message.session_id ?? sessionIdRef.current;
    sessionIdRef.current = sessionId;
    setCurrentSessionId(sessionId);
    setCurrentPaper(paper);
    setPaperTitleMap((prev) => ({ ...prev, [paper.id]: paper.title }));
    setCurrentPaperSource(message.origin ?? null);
    const nextSimpleSearchQuery = message.simple_search_query ?? message.followup_query ?? "";
    const nextKeywordSearchQuery = message.keyword_search_query ?? message.search_keyword ?? message.followup_query ?? "";
    const nextFulltextSearchQuery = message.fulltext_search_query ?? "";
    setSimpleSearchQuery(nextSimpleSearchQuery);
    setKeywordSearchQuery(nextKeywordSearchQuery);
    setFulltextSearchQuery(nextFulltextSearchQuery);
    setSearchModes(message.search_modes ?? []);
    setTrailPaperIds(message.trail_paper_ids ?? []);
    setQueuedPaperIds(message.queued_paper_ids ?? []);
    setNextPaperId(message.next_paper_id ?? null);
    setHits(message.search?.hits ?? []);
    setRejectedCandidates(message.search?.rejected_candidates ?? []);
    setFallbackUsed(Boolean(message.search?.fallback_used));
    setExplanation(message.explanation ?? "");
    const incomingMemo = message.memo ?? "";
    paperMemoRemoteValueRef.current = incomingMemo;
    paperMemoDirtyRef.current = false;
    setPaperMemo(incomingMemo);
    setPaperCosts(message.paper_costs ?? null);
    setSessionCosts(message.session_costs ?? null);
    setBackendNotices(message.notices ?? []);
    setAudioUrls(message.audio_urls && message.audio_urls.length > 0 ? message.audio_urls : message.audio_url ? [message.audio_url] : []);
    setAudioIndex(0);
    setAudioDurationMs(null);
    setIsPlaying(shouldAutoPlayRef.current);
    setLoading(false);
    setPendingAction("idle");
    await Promise.all([refreshHistory(), refreshSessions()]);
  };

  const handleSessionMessage = async (message: SessionEventMessage) => {
    if (typeof message.seq === "number" && message.seq <= lastEventSeqRef.current) {
      return;
    }
    if (typeof message.seq === "number") {
      lastEventSeqRef.current = message.seq;
    }
    if (message.type === "error") {
      setError(message.message ?? "session error");
      setLoading(false);
      setPendingAction("idle");
      return;
    }
    if (message.type === "session_started") {
      setCurrentSessionId(message.session_id ?? null);
      sessionIdRef.current = message.session_id ?? null;
    setPaperCosts(null);
    setSessionCosts(null);
    setBackendNotices([]);
    setQueuedPaperIds([]);
    setNextPaperId(null);
    setActiveTab("session");
    setPendingAction("idle");
      await refreshHistory();
      await refreshSessions();
      return;
    }
    if (message.type === "paper_ready") {
      await applyPaperReadyEvent(message);
      return;
    }
    if (message.type === "session_advanced") {
      await refreshHistory();
      await refreshSessions();
      return;
    }
    if (message.type === "session_queued") {
      setQueuedPaperIds(message.queued_paper_ids ?? []);
      setNextPaperId(message.next_paper_id ?? null);
      await refreshSessions();
      return;
    }
    if (message.type === "session_costs_updated") {
      if (message.session_costs) {
        setSessionCosts(message.session_costs);
      }
      if (currentPaper?.id && message.paper_id === currentPaper.id && message.paper_costs) {
        setPaperCosts(message.paper_costs);
      }
      return;
    }
    if (message.type === "session_regenerated") {
      await refreshHistory();
      await refreshSessions();
      return;
    }
    if (message.type === "session_stopped") {
      setLoading(false);
      setPendingAction("idle");
      stopAudio();
      setCurrentSessionId(null);
      setPaperCosts(null);
      setSessionCosts(null);
      setBackendNotices([]);
      setQueuedPaperIds([]);
      setNextPaperId(null);
      sessionIdRef.current = null;
      setActiveTab("start");
      await refreshHistory();
      await refreshSessions();
    }
  };

  useEffect(() => {
    setMessageHandler(handleSessionMessage);
  }, [handleSessionMessage, setMessageHandler]);

  const handleStart = async () => {
    if (!databaseReady) {
      setError("データベース初期化中です。しばらくしてから再試行してください。");
      return;
    }
    const sourceUrl = form.sourceUrl.trim();
    if (!sourceUrl) {
      setError("開始URLを入力してください。");
      return;
    }
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN && sessionIdRef.current) {
      socketRef.current.send(
        JSON.stringify({
          type: "stop",
          session_id: sessionIdRef.current,
        }),
      );
    }
    sessionIdRef.current = null;
    closeSocket(false);
    sessionIdRef.current = null;
    lastEventSeqRef.current = 0;
    shouldAutoPlayRef.current = true;
    setError(null);
    setLoading(true);
    setPendingAction("start");
    setCurrentSessionId(null);
    setCurrentPaper(null);
    setCurrentPaperSource(null);
    setSimpleSearchQuery("");
    setKeywordSearchQuery("");
    setFulltextSearchQuery("");
    setSearchModes([]);
    setTrailPaperIds([]);
    setQueuedPaperIds([]);
    setPaperTitleMap({});
    setHits([]);
    setRejectedCandidates([]);
    setFallbackUsed(false);
    setExplanation("");
    resetAudio();
    setActiveTab("session");
    openSessionSocket(
      {
        type: "start",
        source_url: sourceUrl,
        model_name: form.modelName,
        include_old_vectors: form.includeOldVectors,
        limit: form.limit,
        route1_weight: form.route1Weight,
        route2_weight: form.route2Weight,
        seed: null,
        search_modes: [
          ...(form.useSimpleSearch ? ["simple"] : []),
          ...(form.useKeywordSearch ? ["keyword_list"] : []),
          ...(form.useFulltextSearch ? ["fulltext_query"] : []),
        ],
      },
      false,
    );
  };

  const handleResumeSession = async (sessionId: string) => {
    if (!databaseReady) {
      setError("データベース初期化中です。しばらくしてから再試行してください。");
      return;
    }
    const normalizedSessionId = sessionId.trim();
    if (!normalizedSessionId) {
      setError("session_id がありません。");
      return;
    }
    setActiveTab("session");
    setLoading(true);
    setPendingAction("resume");
    setBackendNotices([]);
    let snapshot: SessionSnapshot;
    try {
      snapshot = await getSession(normalizedSessionId);
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : "session lookup failed");
      setLoading(false);
      setPendingAction("idle");
      return;
    }
    const eventsResponse = await getSessionEvents(snapshot.session_id, 0).catch((caught: unknown) => {
      setError(caught instanceof Error ? caught.message : "session replay failed");
      return null;
    });
    if (eventsResponse === null) {
      setLoading(false);
      setPendingAction("idle");
      return;
    }
    stopAudio();
    closeSocket(false);
    sessionIdRef.current = snapshot.session_id;
    lastEventSeqRef.current = 0;
    shouldAutoPlayRef.current = false;
    setError(null);
    setPaperCosts(null);
    setSessionCosts(null);
    const replayState = replaySessionEvents(snapshot, eventsResponse.events);
    sessionIdRef.current = replayState.currentSessionId;
    lastEventSeqRef.current = replayState.lastEventSeq;
    setCurrentSessionId(replayState.currentSessionId);
    setCurrentPaper(replayState.currentPaper);
    setCurrentPaperSource(replayState.currentPaperSource);
    setSimpleSearchQuery(replayState.simpleSearchQuery);
    setKeywordSearchQuery(replayState.keywordSearchQuery);
    setFulltextSearchQuery(replayState.fulltextSearchQuery);
    setSearchModes(replayState.searchModes);
    setTrailPaperIds(replayState.trailPaperIds);
    setQueuedPaperIds(replayState.queuedPaperIds);
    setNextPaperId(replayState.nextPaperId);
    setPaperTitleMap(replayState.paperTitleMap);
    setHits(replayState.hits);
    setRejectedCandidates(replayState.rejectedCandidates);
    setFallbackUsed(replayState.fallbackUsed);
    setExplanation(replayState.explanation);
    paperMemoRemoteValueRef.current = replayState.memo;
    paperMemoDirtyRef.current = false;
    setPaperMemo(replayState.memo);
    setPaperCosts(replayState.paperCosts);
    setSessionCosts(replayState.sessionCosts);
    setBackendNotices(replayState.notices);
    setAudioUrls(replayState.audioUrls);
    setAudioIndex(0);
    setAudioDurationMs(replayState.audioDurationMs);
    setIsPlaying(replayState.isPlaying);
    setActiveTab(replayState.activeTab);
    if (replayState.currentSessionId !== null) {
      openSessionSocket(undefined, false);
    }
    setLoading(false);
  };

  const handleToggleFavorite = async (paperId: string) => {
    try {
      const response = await toggleFavorite(paperId);
      if (response.favorited) {
        const favoriteResponse = await listFavorites();
        setFavorites(favoriteResponse.items);
        return;
      }
      setFavorites((current) => current.filter((item) => item.paper_id !== paperId));
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : "favorite update failed");
    }
  };

  const handleStop = () => {
    stopAudio();
  };

  const handleResume = () => {
    if (loading) {
      setError("処理中です。しばらくしてから再開してください。");
      return;
    }
    if (!currentSessionId || audioUrls.length === 0) {
      setError("再開する再生がありません。");
      return;
    }
    setError(null);
    shouldAutoPlayRef.current = true;
    setBackendNotices([]);
    const current = audioRef.current;
    if (current && current.duration > 0 && current.currentTime >= current.duration - 0.1) {
      current.currentTime = 0;
    }
    if (current) {
      current.muted = false;
      current.volume = Math.min(1, Math.max(0, audioVolume));
    }
    setIsPlaying(true);
    if (current) {
      void current.play().catch(() => {
        // クリック操作後でも失敗する場合は state 側の effect に任せる
      });
    }
  };

  useLayoutEffect(() => {
    resumeAudioRef.current = handleResume;
    pauseAudioRef.current = pauseAudio;
    stopAudioRef.current = stopAudio;
  }, [handleResume, pauseAudio, stopAudio]);

  useLayoutEffect(() => {
    audioUrlsLengthRef.current = audioUrls.length;
  }, [audioUrls.length]);

  const handleNext = () => {
    if (!databaseReady) {
      setError("データベース初期化中です。しばらくしてから再試行してください。");
      return;
    }
    if (!currentSessionId || socketRef.current?.readyState !== WebSocket.OPEN) {
      setError("次へ進める session がありません。");
      return;
    }
    stopAudio();
    shouldAutoPlayRef.current = true;
    setBackendNotices([]);
    setError(null);
    setLoading(true);
    setPendingAction("next");
    socketRef.current.send(
      JSON.stringify({
        type: "next",
        session_id: currentSessionId,
      }),
    );
  };

  const handleRegenerate = () => {
    if (!databaseReady) {
      setError("データベース初期化中です。しばらくしてから再試行してください。");
      return;
    }
    if (!currentSessionId || socketRef.current?.readyState !== WebSocket.OPEN) {
      setError("再生成する session がありません。");
      return;
    }
    stopAudio();
    shouldAutoPlayRef.current = true;
    setBackendNotices([]);
    setError(null);
    setLoading(true);
    setPendingAction("regenerate");
    socketRef.current.send(
      JSON.stringify({
        type: "regenerate",
        session_id: currentSessionId,
      }),
    );
  };

  const handleAudioEnded = () => {
    if (audioIndex < audioUrls.length - 1) {
      setAudioIndex((current) => current + 1);
      return;
    }
    if (!currentSessionId || socketRef.current?.readyState !== WebSocket.OPEN) {
      return;
    }
    handleNext();
  };

  useLayoutEffect(() => {
    if (!("mediaSession" in navigator)) {
      return;
    }
    const mediaSession = navigator.mediaSession;
    const handlePlayAction = () => {
      if (audioUrlsLengthRef.current > 0) {
        resumeAudioRef.current();
      }
    };
    const handlePauseAction = () => {
      pauseAudioRef.current();
    };
    const handleStopAction = () => {
      stopAudioRef.current();
    };
    mediaSession.setActionHandler("play", handlePlayAction);
    mediaSession.setActionHandler("pause", handlePauseAction);
    mediaSession.setActionHandler("stop", handleStopAction);
    mediaSession.setActionHandler("previoustrack", null);
    mediaSession.setActionHandler("nexttrack", null);
    return () => {
      mediaSession.setActionHandler("play", null);
      mediaSession.setActionHandler("pause", null);
      mediaSession.setActionHandler("stop", null);
      mediaSession.setActionHandler("previoustrack", null);
      mediaSession.setActionHandler("nexttrack", null);
    };
  }, []);

  useEffect(() => {
    if (!("mediaSession" in navigator)) {
      return;
    }
    const mediaSession = navigator.mediaSession;
    if (typeof MediaMetadata !== "undefined" && currentPaper !== null) {
      mediaSession.metadata = new MediaMetadata({
        title: currentPaper.title,
        artist: currentPaper.authors ?? "quick-auditory-learning",
        album: currentPaper.id,
      });
    } else {
      mediaSession.metadata = null;
    }
    mediaSession.playbackState = audioUrls.length > 0 && isPlaying ? "playing" : "paused";
  }, [audioUrls.length, currentPaper, isPlaying]);

  return (
    <main className="app-shell">
      <section className="app-frame">
        <header className="app-topbar">
          <div className="app-brand-row">
            <p className="app-title">quick-auditory-learning</p>
            <p className="app-status">{headerStatus}</p>
          </div>
          <nav className="tab-bar" aria-label="画面切り替え">
            <button
              type="button"
              className={`tab-button${activeTab === "start" ? " is-active" : ""}`}
              onClick={() => setActiveTab("start")}
            >
              開始・続きから
            </button>
            <button
              type="button"
              className={`tab-button${activeTab === "favorites" ? " is-active" : ""}`}
              onClick={() => setActiveTab("favorites")}
            >
              お気に入り
            </button>
            {showSessionTab ? (
              <button
                type="button"
                className={`tab-button${activeTab === "session" ? " is-active" : ""}`}
                onClick={() => setActiveTab("session")}
              >
                現在のセッション
              </button>
            ) : null}
          </nav>
        </header>

        {error ? (
          <section className="card error-card" role="alert">
            <h2>エラー</h2>
            <p className="error">{error}</p>
            <p className="muted">backend.log を確認してください。必要なら `/health` で backend の生存確認をしてください。</p>
          </section>
        ) : null}

        {backendNotices.length > 0 ? (
          <section className="card notice-card" role="status" aria-live="polite">
            <h2>通知</h2>
            <ul className="notice-list">
              {backendNotices.map((notice, index) => (
                <li key={`${index}-${notice}`}>{notice}</li>
              ))}
            </ul>
          </section>
        ) : null}

        {audioUrls.length > 0 ? (
          <div className="audio-surface" aria-hidden="true">
            <audio
              ref={audioRef}
              preload="auto"
              src={resolveAudioSourceUrl(apiBaseUrl, audioUrls[audioIndex])}
              className="audio-player"
              onPlay={() => {
                setIsPlaying(true);
              }}
              onPause={(event) => {
                if (event.currentTarget.ended) {
                  return;
                }
                setIsPlaying(false);
              }}
              onLoadedMetadata={(event) => {
                const duration = event.currentTarget.duration;
                if (Number.isFinite(duration) && duration > 0) {
                  setAudioDurationMs(Math.round(duration * 1000));
                }
              }}
              onEnded={handleAudioEnded}
            />
          </div>
        ) : null}

        <div className="tab-panels">
          {activeTab === "start" ? (
            <section className="tab-layout">
              <section className="card start-card">
                <div className="card-head start-card-head">
                  <div>
                    <h2>はじめから</h2>
                  </div>
                  {!databaseReady ? <p className="meta start-status">データベース初期化中です。</p> : null}
                </div>

                <div className="start-line">
                  <button
                    type="button"
                    className="start-button"
                    onClick={() => void handleStart()}
                    disabled={!databaseReady || (loading && wsStatus === "connecting")}
                  >
                    {loading ? "処理中..." : "開始"}
                  </button>
                  <input
                    value={form.sourceUrl}
                    onChange={(event) => setForm({ ...form, sourceUrl: event.target.value })}
                    placeholder="arXivのURLを入力"
                  />
                </div>

                <details className="search-details">
                  <summary>検索詳細設定</summary>
                  <div className="panel search-panel">
                    <div className="search-mode-grid">
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={form.useSimpleSearch}
                          onChange={(event) => setForm({ ...form, useSimpleSearch: event.target.checked })}
                        />
                        通常検索
                      </label>
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={form.useKeywordSearch}
                          onChange={(event) => setForm({ ...form, useKeywordSearch: event.target.checked })}
                        />
                        キーワード列
                      </label>
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={form.useFulltextSearch}
                          onChange={(event) => setForm({ ...form, useFulltextSearch: event.target.checked })}
                        />
                        全文検索クエリ
                      </label>
                    </div>
                    <label className="checkbox">
                      <input
                        type="checkbox"
                        checked={form.includeOldVectors}
                        onChange={(event) => setForm({ ...form, includeOldVectors: event.target.checked })}
                      />
                      古いベクトルも検索
                    </label>
                    <div className="grid-2">
                      <label>
                        件数
                        <input
                          type="number"
                          value={form.limit}
                          min={1}
                          max={50}
                          onChange={(event) => setForm({ ...form, limit: Number(event.target.value) })}
                        />
                      </label>
                      <label>
                        ルート1重み
                        <input
                          type="number"
                          step="0.05"
                          min={0}
                          max={1}
                          value={form.route1Weight}
                          onChange={(event) => setForm({ ...form, route1Weight: Number(event.target.value) })}
                        />
                      </label>
                    </div>
                  </div>
                </details>
              </section>

              <section className="card">
                <div className="card-head session-list-head">
                  <div>
                    <h2>続きから</h2>
                    <p className="meta">最新 {visibleSessionSummaries.length} 件 / 全 {sessionSummaries.length} 件</p>
                  </div>
                  <button
                    type="button"
                    className="session-list-toggle"
                    onClick={() => setShowAllSessions((current) => !current)}
                    disabled={sessionSummaries.length === 0}
                  >
                    {showAllSessions ? "最新 1 件に戻す" : "全部表示する"}
                  </button>
                </div>
                {visibleSessionSummaries.length > 0 ? (
                  <ul className="session-list">
                    {visibleSessionSummaries.map((session) => {
                      const isCurrent = session.session_id === currentSessionId;
                      const updatedAt = new Date(session.updated_at).toLocaleString("ja-JP", {
                        dateStyle: "short",
                        timeStyle: "short",
                      });
                      return (
                        <li key={session.session_id} className="session-row">
                          <button
                            type="button"
                            className={`session-open-button${isCurrent ? " is-current" : ""}`}
                            onClick={() => {
                              if (isCurrent) {
                                setActiveTab("session");
                                return;
                              }
                              void handleResumeSession(session.session_id);
                            }}
                            disabled={!databaseReady}
                          >
                            {isCurrent ? "再生中" : "再開"}
                          </button>
                          <div className={`session-item${isCurrent ? " is-current" : ""}`}>
                            <div className="session-item-main">
                              <h3>{updatedAt}</h3>
                              <p className="meta">session {session.session_id} / {session.status}</p>
                              <p className="meta">
                                合計処理時間 {formatDurationSeconds(session.total_wall_elapsed_ms)} / {formatUsd(session.total_generation_cost_usd)}
                              </p>
                            </div>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <p className="muted">再開できるセッションはまだありません。</p>
                )}
              </section>
            </section>
          ) : null}

          {activeTab === "favorites" ? (
            <section className="tab-layout">
              <section className="card">
                <div className="card-head">
                  <div>
                    <p className="eyebrow">favorites</p>
                    <h2>お気に入り確認・管理</h2>
                  </div>
                  <p className="meta">登録済み {favorites.length} 件</p>
                </div>
                {favorites.length === 0 ? (
                  <p className="muted">まだありません。</p>
                ) : (
                  <ul className="favorites-list">
                    {favorites.map((favorite) => (
                      <li key={favorite.paper_id} className="favorites-item">
                        <div className="favorites-item-main">
                          <span className="favorites-paper-id">{favorite.paper_id}</span>
                          <a
                            className="favorites-paper-link"
                            href={`https://arxiv.org/abs/${encodeURIComponent(favorite.paper_id)}`}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {favorite.title}
                          </a>
                        </div>
                        <button type="button" onClick={() => void handleToggleFavorite(favorite.paper_id)}>
                          解除
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </section>
          ) : null}

          {activeTab === "session" && showSessionTab ? (
            <section className="tab-layout">
              <section className={`card player-card${loading && pendingAction === "regenerate" ? " is-regenerating" : ""}`}>
                {currentPaper ? (
                  <div className="current-session-body">
                    <div className="current-session-actions" aria-label="現在のセッション操作">
                      <div className="current-session-action-group" role="group" aria-label="再生操作">
                        <button
                          type="button"
                          className="current-session-action-button"
                          onClick={handleStop}
                          disabled={audioUrls.length === 0 || !isPlaying}
                          aria-label="再生停止"
                          title="再生停止"
                        >
                          <SessionActionIcon kind="stop" />
                        </button>
                        <button
                          type="button"
                          className="current-session-action-button"
                          onClick={handleResume}
                          disabled={audioUrls.length === 0 || isPlaying || loading}
                          aria-label="再生再開"
                          title="再生再開"
                        >
                          <SessionActionIcon kind="play" />
                        </button>
                        <button
                          type="button"
                          className="current-session-action-button"
                          onClick={handleNext}
                          disabled={!currentSessionId || wsStatus !== "connected" || loading}
                          aria-label="次へ進む"
                          title="次へ進む"
                        >
                          <SessionActionIcon kind="next" />
                        </button>
                        <button
                          type="button"
                          className={`current-session-action-button${loading && pendingAction === "regenerate" ? " is-loading" : ""}`}
                          onClick={handleRegenerate}
                          disabled={!currentSessionId || wsStatus !== "connected" || loading}
                          aria-label={loading ? "再生成中" : "再生成"}
                          title={loading ? "再生成中" : "再生成"}
                        >
                          <SessionActionIcon kind="regenerate" />
                        </button>
                      </div>
                      <div className="audio-controls" role="group" aria-label="音声設定">
                        <label className="audio-control-label">
                          <span>音量</span>
                          <input
                            type="range"
                            min={0}
                            max={3}
                            step={0.05}
                            value={audioVolume}
                            onChange={(e) => setAudioVolume(parseFloat(e.target.value))}
                            aria-label="音量"
                          />
                          <span className="audio-control-value">{Math.round(audioVolume * 100)}%</span>
                        </label>
                        <label className="audio-control-label">
                          <span>速度</span>
                          <input
                            type="range"
                            min={0.5}
                            max={3}
                            step={0.25}
                            value={audioRate}
                            onChange={(e) => setAudioRate(parseFloat(e.target.value))}
                            aria-label="再生速度"
                          />
                          <span className="audio-control-value">{audioRate.toFixed(2)}x</span>
                        </label>
                      </div>
                      <button
                        type="button"
                        className={`current-session-action-button favorite-action${favoriteSet.has(currentPaper.id) ? " is-active" : ""}`}
                        onClick={() => void handleToggleFavorite(currentPaper.id)}
                        aria-label={favoriteSet.has(currentPaper.id) ? "お気に入り解除" : "お気に入り"}
                        title={favoriteSet.has(currentPaper.id) ? "お気に入り解除" : "お気に入り"}
                      >
                        <SessionActionIcon kind="favorite" />
                      </button>
                    </div>

                    <article className="current-paper">
                      {loading && pendingAction === "regenerate" ? (
                        <div className="regenerate-status" role="status" aria-live="polite">
                          <span className="regenerate-spinner" aria-hidden="true" />
                          <div>
                            <p className="eyebrow">再生成中</p>
                            <p>再生成中です。音声と解説を作り直しています。</p>
                          </div>
                        </div>
                      ) : null}
                      <p className="paper-id">{currentPaper.id}</p>
                      <h3>{currentPaper.title}</h3>
                      {explanation ? <p className="explanation">{explanation}</p> : null}
                      <section className="memo-section" aria-label="メモ">
                        <div className="memo-head">
                          <h3>メモ</h3>
                        </div>
                        <textarea
                          className="memo-input"
                          value={paperMemo}
                          disabled={loading}
                          onChange={(event) => {
                            setPaperMemo(event.target.value);
                          }}
                          placeholder={loading ? "読み込み中..." : "気づき、疑問、後で読む観点など"}
                          rows={5}
                        />
                        <p className="memo-sync" aria-live="polite">
                          {paperMemoNotice || ""}
                        </p>
                        <p className="memo-status" aria-live="polite">
                          {paperMemoStatus === "saving"
                            ? "保存中"
                            : paperMemoStatus === "error"
                                ? "保存に失敗しました"
                                : ""}
                        </p>
                      </section>
                      <details className="session-details">
                        <summary>詳細を表示する</summary>
                        <div className="session-detail-stack">
                          <section className="session-detail-section">
                            <h3>状態</h3>
                            <dl className="session-detail-list">
                              <div className="session-detail-row">
                                <dt>session</dt>
                                <dd>{currentSessionId ?? "none"}</dd>
                              </div>
                              <div className="session-detail-row">
                                <dt>ws</dt>
                                <dd>{wsStatus}</dd>
                              </div>
                              <div className="session-detail-row">
                                <dt>trail</dt>
                                <dd>{trailPaperIds.length} 件</dd>
                              </div>
                              <div className="session-detail-row">
                                <dt>結果</dt>
                                <dd>{hits.length} 件</dd>
                              </div>
                              <div className="session-detail-row">
                                <dt>通常検索語</dt>
                                <dd>{simpleSearchQuery || "unknown"}</dd>
                              </div>
                              <div className="session-detail-row">
                                <dt>キーワード列</dt>
                                <dd>{keywordSearchQuery || "unknown"}</dd>
                              </div>
                              <div className="session-detail-row">
                                <dt>全文検索クエリ</dt>
                                <dd>{fulltextSearchQuery || "unknown"}</dd>
                              </div>
                              <div className="session-detail-row">
                                <dt>補足</dt>
                                <dd>{fallbackUsed ? "一致が弱いためランダム選択しています。" : "通常検索"}</dd>
                              </div>
                              <div className="session-detail-row">
                                <dt>検索方式</dt>
                                <dd>{searchModes.length > 0 ? searchModes.join(" / ") : "unknown"}</dd>
                              </div>
                              <div className="session-detail-row">
                                <dt>埋め込みモデル</dt>
                                <dd>{form.modelName}</dd>
                              </div>
                            </dl>
                          </section>

                          <section className="session-detail-section">
                            <div className="cost-tab-header">
                              <div className="cost-tab-copy">
                                <h3>コスト</h3>
                              </div>
                              <div className="cost-tab-bar" role="tablist" aria-label="コストの表示単位">
                                <button
                                  type="button"
                                  role="tab"
                                  aria-selected={costTab === "paper"}
                                  className={`cost-tab-button${costTab === "paper" ? " is-active" : ""}`}
                                  onClick={() => setCostTab("paper")}
                                >
                                  再生単位
                                </button>
                                <button
                                  type="button"
                                  role="tab"
                                  aria-selected={costTab === "session"}
                                  className={`cost-tab-button${costTab === "session" ? " is-active" : ""}`}
                                  onClick={() => setCostTab("session")}
                                >
                                  セッション単位
                                </button>
                              </div>
                            </div>
                            {costTab === "paper" ? (
                              paperCosts ? (
                                <>
                                  <table className="session-cost-table">
                                    <colgroup>
                                      <col className="session-cost-table-kind-col" />
                                      <col className="session-cost-table-value-col" />
                                      <col className="session-cost-table-value-col" />
                                      <col className="session-cost-table-value-col" />
                                    </colgroup>
                                    <thead>
                                      <tr>
                                        <th>種類</th>
                                        <th>時間</th>
                                        <th>処理待ち時間</th>
                                        <th>費用</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {buildCostRows(paperCosts.items).map((row) => (
                                        <tr key={row.kind}>
                                          <td>{row.kind}</td>
                                          <td>{row.elapsedLabel}</td>
                                          <td>{row.elapsedWithoutPrefetchLabel}</td>
                                          <td>{row.costLabel}</td>
                                        </tr>
                                      ))}
                                    </tbody>
                                    <tfoot>
                                      <tr>
                                        <td>合計</td>
                                        <td>{formatOptionalCostValue(paperCosts.total_elapsed_ms, formatDurationSeconds)}</td>
                                        <td>{formatOptionalCostValue(paperCosts.total_elapsed_ms_without_prefetch, formatDurationSeconds)}</td>
                                        <td>{formatOptionalCostValue(paperCosts.total_cost_usd, formatUsd)}</td>
                                      </tr>
                                    </tfoot>
                                  </table>
                                  <p className="cost-audio-duration">{formatAudioDurationNote(paperCosts, audioDurationMs)}</p>
                                </>
                              ) : (
                                <p className="muted">まだありません。</p>
                              )
                            ) : (
                              sessionCosts ? (
                                <>
                                  <table className="session-cost-table">
                                    <colgroup>
                                      <col className="session-cost-table-kind-col" />
                                      <col className="session-cost-table-value-col" />
                                      <col className="session-cost-table-value-col" />
                                      <col className="session-cost-table-value-col" />
                                    </colgroup>
                                    <thead>
                                      <tr>
                                        <th>種類</th>
                                        <th>時間</th>
                                        <th>処理待ち時間</th>
                                        <th>費用</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {buildCostRows(sessionCosts.items).map((row) => (
                                        <tr key={row.kind}>
                                          <td>{row.kind}</td>
                                          <td>{row.elapsedLabel}</td>
                                          <td>{row.elapsedWithoutPrefetchLabel}</td>
                                          <td>{row.costLabel}</td>
                                        </tr>
                                      ))}
                                    </tbody>
                                    <tfoot>
                                      <tr>
                                        <td>合計</td>
                                        <td>{formatOptionalCostValue(sessionCosts.total_elapsed_ms, formatDurationSeconds)}</td>
                                        <td>{formatOptionalCostValue(sessionCosts.total_elapsed_ms_without_prefetch, formatDurationSeconds)}</td>
                                        <td>{formatOptionalCostValue(sessionCosts.total_cost_usd, formatUsd)}</td>
                                      </tr>
                                    </tfoot>
                                  </table>
                                  <p className="cost-audio-duration">{formatAudioDurationNote(sessionCosts, audioDurationMs)}</p>
                                </>
                              ) : (
                                <p className="muted">まだありません。</p>
                              )
                            )}
                          </section>

                          <section className="session-detail-section">
                            <h3>入力URL</h3>
                            <dl className="session-detail-list">
                              <div className="session-detail-row">
                                <dt>URL</dt>
                                <dd>{form.sourceUrl}</dd>
                              </div>
                            </dl>
                          </section>

                          <section className="session-detail-section">
                            <h3>取得経路</h3>
                            <dl className="session-detail-list">
                              <div className="session-detail-row">
                                <dt>origin</dt>
                                <dd>{formatPaperOrigin(currentPaperSource)}</dd>
                              </div>
                            </dl>
                          </section>

                          <section className="session-detail-section">
                            <h3>再生履歴</h3>
                            {trailPaperIds.length === 0 ? (
                              <p className="muted">まだ履歴はありません。</p>
                            ) : (
                              <ol className="trail-list">
                                {[...trailPaperIds].reverse().map((id) => (
                                  <li key={id} className="trail-item">
                                    <span className="trail-item-id">{id}</span>
                                    {paperTitleMap[id] ? (
                                      <span className="trail-item-title">{paperTitleMap[id]}</span>
                                    ) : null}
                                  </li>
                                ))}
                              </ol>
                            )}
                          </section>

                          <section className="session-detail-section">
                            <h3>原文サマリー</h3>
                            <p>{currentPaper.abstract}</p>
                          </section>

                        </div>
                      </details>
                    </article>
                  </div>
                ) : loading ? (
                  <div className="regenerate-empty-state" role="status" aria-live="polite">
                    <span className="regenerate-spinner" aria-hidden="true" />
                    <div>
                      <p className="eyebrow">{pendingAction === "regenerate" ? "再生成中" : "開始準備中"}</p>
                      <p>
                        {pendingAction === "regenerate"
                          ? "再生成中です。しばらくお待ちください。"
                          : "開始中です。セッションの準備ができるまでお待ちください。"}
                      </p>
                    </div>
                  </div>
                ) : (
                  <p className="muted">開始URLを入力して session を開始してください。</p>
                )}
              </section>

              <section className="card">
                <h2>検索結果</h2>
                {hits.length === 0 ? (
                  <p className="muted">まだ結果はありません。</p>
                ) : (
                  <SearchResultList
                    items={hits.map((hit) => {
                      const state = buildSearchResultState({
                        currentPaperId: currentPaper?.id ?? null,
                        paperId: hit.paper.id,
                        selectedNextPaperId,
                        trailPaperIds: trailPaperSet,
                        favoritePaperIds: favoriteSet,
                        canInteract: Boolean(currentSessionId && socketRef.current?.readyState === WebSocket.OPEN && !loading),
                      });
                      const isExplicitQueued = queuedPaperSet.has(hit.paper.id);
                      const sourceModes = uniqueSearchModes(hit.source_modes).map((mode) => formatSearchMode(mode));
                      return {
                        id: hit.paper.id,
                        paperIdLabel: hit.paper.id,
                        title: hit.paper.title,
                        meta: (
                          <>
                            <span className="paper-categories">{formatCategories(hit.paper.categories)}</span>
                            score {hit.score.toFixed(4)} / route1 {hit.route1_score.toFixed(4)} / route2 {hit.route2_score.toFixed(4)}
                          </>
                        ),
                        sourceModes,
                        isSelected: state.isSelected,
                        isQueued: state.isQueued,
                        isReplayed: state.isReplayed,
                        isFavorite: state.isFavorite,
                        canInteract: state.canInteract,
                        onToggleQueue: () => {
                          if (!currentSessionId || socketRef.current?.readyState !== WebSocket.OPEN) return;
                          setError(null);
                          socketRef.current.send(
                            JSON.stringify({
                              type: isExplicitQueued ? "dequeue" : "queue",
                              session_id: currentSessionId,
                              paper_id: hit.paper.id,
                            }),
                          );
                        },
                        onToggleFavorite: () => void handleToggleFavorite(hit.paper.id),
                      };
                    })}
                  />
                )}
              </section>

              <div className="layout-grid">
                <section className="card full-width-card">
                  <h2>前の論文から検索した他の論文</h2>
                  {rejectedCandidates.length === 0 ? (
                    <p className="muted">まだありません。</p>
                ) : (
                  <SearchResultList
                    items={rejectedCandidates.map((candidate) => {
                      const state = buildSearchResultState({
                        currentPaperId: currentPaper?.id ?? null,
                        paperId: candidate.paper_id,
                        selectedNextPaperId,
                        trailPaperIds: trailPaperSet,
                        favoritePaperIds: favoriteSet,
                        canInteract: Boolean(currentSessionId && socketRef.current?.readyState === WebSocket.OPEN && !loading),
                      });
                      const isExplicitQueued = queuedPaperSet.has(candidate.paper_id);
                      const sourceModes = uniqueSearchModes(candidate.source_modes).map((mode) => formatSearchMode(mode));
                      return {
                        id: candidate.paper_id,
                        paperIdLabel: candidate.paper_id,
                        title: candidate.title,
                        meta: (
                          <>
                            score {candidate.score.toFixed(4)} / {candidate.reason}
                          </>
                        ),
                        sourceModes,
                        isSelected: state.isSelected,
                        isQueued: state.isQueued,
                        isReplayed: state.isReplayed,
                        isFavorite: state.isFavorite,
                        canInteract: state.canInteract,
                        onToggleQueue: () => {
                          if (!currentSessionId || socketRef.current?.readyState !== WebSocket.OPEN) return;
                          setError(null);
                          socketRef.current.send(
                            JSON.stringify({
                              type: isExplicitQueued ? "dequeue" : "queue",
                              session_id: currentSessionId,
                              paper_id: candidate.paper_id,
                            }),
                          );
                        },
                        onToggleFavorite: () => void handleToggleFavorite(candidate.paper_id),
                      };
                    })}
                  />
                )}
                </section>

              </div>
            </section>
          ) : null}
        </div>
      </section>
    </main>
  );
}
