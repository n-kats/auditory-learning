import { useEffect, useMemo, useRef, useState } from "react";
import type { FavoritePaperItem, Paper, SearchCandidate, SearchHit, SessionCosts } from "./api";
import { applyReplayToAppSessionState, emptyAppSessionState, type AppSessionState } from "./appSessionState";
import type { SessionViewState } from "./sessionViewState";
import type { SessionReplayState } from "./sessionReplay";

type UseAppSessionStateResult = {
  currentSessionId: string | null;
  currentPaper: Paper | null;
  currentPaperSource: string | null;
  simpleSearchQuery: string;
  keywordSearchQuery: string;
  fulltextSearchQuery: string;
  searchModes: string[];
  trailPaperIds: string[];
  nextPaperId: string | null;
  searchPaperId: string | null;
  hits: SearchHit[];
  rejectedCandidates: SearchCandidate[];
  previousSearchPaperId: string | null;
  previousRejectedCandidates: SearchCandidate[];
  fallbackUsed: boolean;
  explanation: string;
  paperCosts: SessionCosts | null;
  sessionCosts: SessionCosts | null;
  paperTitleMap: Record<string, string>;
  backendNotices: string[];
  setPaperCosts: (value: SessionCosts | null) => void;
  setSessionCosts: (value: SessionCosts | null) => void;
  setBackendNotices: (value: string[]) => void;
  currentAppSessionState: AppSessionState;
  currentAppSessionStateRef: React.MutableRefObject<AppSessionState>;
  applySessionViewState: (nextState: SessionViewState) => void;
  applyAppSessionState: (nextState: AppSessionState) => void;
  applyReplayToState: (replayState: SessionReplayState) => AppSessionState;
};

export function useAppSessionState(
  favorites: FavoritePaperItem[],
  audioState: { audioUrls: string[]; audioIndex: number; audioDurationMs: number | null },
): UseAppSessionStateResult {
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentPaper, setCurrentPaper] = useState<Paper | null>(null);
  const [currentPaperSource, setCurrentPaperSource] = useState<string | null>(null);
  const [simpleSearchQuery, setSimpleSearchQuery] = useState<string>("");
  const [keywordSearchQuery, setKeywordSearchQuery] = useState<string>("");
  const [fulltextSearchQuery, setFulltextSearchQuery] = useState<string>("");
  const [searchModes, setSearchModes] = useState<string[]>([]);
  const [trailPaperIds, setTrailPaperIds] = useState<string[]>([]);
  const [nextPaperId, setNextPaperId] = useState<string | null>(null);
  const [searchPaperId, setSearchPaperId] = useState<string | null>(null);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [rejectedCandidates, setRejectedCandidates] = useState<SearchCandidate[]>([]);
  const [previousSearchPaperId, setPreviousSearchPaperId] = useState<string | null>(null);
  const [previousRejectedCandidates, setPreviousRejectedCandidates] = useState<SearchCandidate[]>([]);
  const [fallbackUsed, setFallbackUsed] = useState(false);
  const [explanation, setExplanation] = useState<string>("");
  const [paperCosts, setPaperCosts] = useState<SessionCosts | null>(null);
  const [sessionCosts, setSessionCosts] = useState<SessionCosts | null>(null);
  const [paperTitleMap, setPaperTitleMap] = useState<Record<string, string>>({});
  const [backendNotices, setBackendNotices] = useState<string[]>([]);
  const currentAppSessionStateRef = useRef<AppSessionState>(emptyAppSessionState());

  const currentAppSessionState = useMemo<AppSessionState>(
    () => ({
      currentSessionId,
      currentPaper,
      currentPaperSource,
      simpleSearchQuery,
      keywordSearchQuery,
      fulltextSearchQuery,
      searchModes,
      trailPaperIds,
      nextPaperId,
      searchPaperId,
      hits,
      rejectedCandidates,
      previousSearchPaperId,
      previousRejectedCandidates,
      fallbackUsed,
      explanation,
      paperCosts,
      sessionCosts,
      paperTitleMap,
      backendNotices,
      audioUrls: audioState.audioUrls,
      audioIndex: audioState.audioIndex,
      audioDurationMs: audioState.audioDurationMs,
      favorites,
    }),
    [
      audioState.audioDurationMs,
      audioState.audioIndex,
      audioState.audioUrls,
      backendNotices,
      currentPaper,
      currentPaperSource,
      currentSessionId,
      explanation,
      fallbackUsed,
      favorites,
      fulltextSearchQuery,
      hits,
      keywordSearchQuery,
      nextPaperId,
      paperCosts,
      paperTitleMap,
      rejectedCandidates,
      previousRejectedCandidates,
      previousSearchPaperId,
      searchModes,
      searchPaperId,
      sessionCosts,
      simpleSearchQuery,
      trailPaperIds,
    ],
  );

  useEffect(() => {
    currentAppSessionStateRef.current = currentAppSessionState;
  }, [currentAppSessionState]);

  const applySessionViewState = (nextState: SessionViewState) => {
    currentAppSessionStateRef.current = {
      ...currentAppSessionStateRef.current,
      ...nextState,
    };
    setCurrentSessionId(nextState.currentSessionId);
    setCurrentPaper(nextState.currentPaper);
    setCurrentPaperSource(nextState.currentPaperSource);
    setSimpleSearchQuery(nextState.simpleSearchQuery);
    setKeywordSearchQuery(nextState.keywordSearchQuery);
    setFulltextSearchQuery(nextState.fulltextSearchQuery);
    setSearchModes(nextState.searchModes);
    setTrailPaperIds(nextState.trailPaperIds);
    setNextPaperId(nextState.nextPaperId);
    setSearchPaperId(nextState.searchPaperId);
    setHits(nextState.hits);
    setRejectedCandidates(nextState.rejectedCandidates);
    setPreviousSearchPaperId(nextState.previousSearchPaperId);
    setPreviousRejectedCandidates(nextState.previousRejectedCandidates);
    setFallbackUsed(nextState.fallbackUsed);
  };

  const applyAppSessionState = (nextState: AppSessionState) => {
    currentAppSessionStateRef.current = nextState;
    applySessionViewState(nextState);
    setExplanation(nextState.explanation);
    setPaperCosts(nextState.paperCosts);
    setSessionCosts(nextState.sessionCosts);
    setBackendNotices(nextState.backendNotices);
    setPaperTitleMap(nextState.paperTitleMap);
  };

  const applyReplayToState = (replayState: SessionReplayState): AppSessionState => {
    const nextState = applyReplayToAppSessionState(currentAppSessionStateRef.current, replayState);
    applyAppSessionState(nextState);
    return nextState;
  };

  return {
    currentSessionId,
    currentPaper,
    currentPaperSource,
    simpleSearchQuery,
    keywordSearchQuery,
    fulltextSearchQuery,
    searchModes,
    trailPaperIds,
    nextPaperId,
    searchPaperId,
    hits,
    rejectedCandidates,
    previousSearchPaperId,
    previousRejectedCandidates,
    fallbackUsed,
    explanation,
    paperCosts,
    sessionCosts,
    paperTitleMap,
    backendNotices,
    setPaperCosts,
    setSessionCosts,
    setBackendNotices,
    currentAppSessionState,
    currentAppSessionStateRef,
    applySessionViewState,
    applyAppSessionState,
    applyReplayToState,
  };
}
