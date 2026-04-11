import { useEffect, useRef, useState } from "react";

import { getHealth, listFavorites, recentHistory, recentSessions, type FavoritePaperItem, type SessionSummary } from "./api";
import { HEALTH_POLL_INTERVAL_MS, SESSION_REFRESH_INTERVAL_MS } from "./backendDirectoryPolling";

type UseBackendDirectoryDataOptions = {
  onError: (message: string) => void;
  onSuccess: () => void;
};

export function useBackendDirectoryData({ onError, onSuccess }: UseBackendDirectoryDataOptions) {
  const [databaseReady, setDatabaseReady] = useState(false);
  const [favorites, setFavorites] = useState<FavoritePaperItem[]>([]);
  const [history, setHistory] = useState<Array<{ from_paper_id: string | null; to_paper_id: string }>>([]);
  const [sessionSummaries, setSessionSummaries] = useState<SessionSummary[]>([]);
  const onErrorRef = useRef(onError);
  const onSuccessRef = useRef(onSuccess);

  useEffect(() => {
    onErrorRef.current = onError;
    onSuccessRef.current = onSuccess;
  }, [onError, onSuccess]);

  const refreshFavorites = async () => {
    const response = await listFavorites();
    setFavorites(response.items);
  };

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
        onSuccessRef.current();
      } catch (caught: unknown) {
        if (cancelled) {
          return;
        }
        onErrorRef.current(caught instanceof Error ? caught.message : "failed to load initial data");
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
        onSuccessRef.current();
      } catch (caught: unknown) {
        if (cancelled) {
          return;
        }
        setDatabaseReady(false);
        onErrorRef.current(caught instanceof Error ? caught.message : "health check failed");
      }
    };
    void pollHealth();
    const timer = window.setInterval(() => {
      void pollHealth();
    }, HEALTH_POLL_INTERVAL_MS);
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
    }, SESSION_REFRESH_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [databaseReady]);

  return {
    databaseReady,
    setDatabaseReady,
    favorites,
    setFavorites,
    history,
    setHistory,
    sessionSummaries,
    setSessionSummaries,
    refreshFavorites,
    refreshHistory,
    refreshSessions,
  };
}
