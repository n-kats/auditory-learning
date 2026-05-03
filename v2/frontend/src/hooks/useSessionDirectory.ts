import { useEffect, useState } from "react";

import { fetchSessions, type SessionSummary } from "../api";

type UseSessionDirectoryResult = {
  sessions: SessionSummary[];
  isLoading: boolean;
  error: string | null;
  refreshSessions: () => Promise<void>;
};

export function useSessionDirectory(): UseSessionDirectoryResult {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshSessions = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const nextSessions = await fetchSessions();
      setSessions(nextSessions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "セッション一覧の取得に失敗しました。");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void refreshSessions();
  }, []);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      void refreshSessions();
    }, 15000);

    return () => window.clearInterval(intervalId);
  }, []);

  return {
    sessions,
    isLoading,
    error,
    refreshSessions,
  };
}
