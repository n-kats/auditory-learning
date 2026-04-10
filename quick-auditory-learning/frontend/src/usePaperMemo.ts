import { useEffect, useRef, useState } from "react";

import { getPaperMemo, savePaperMemo, toWebSocketUrl } from "./api";
import { normalizeMemoText, shouldSaveMemo } from "./paperMemo";

export function usePaperMemo(currentPaperId: string | null, onError?: (message: string) => void) {
  const [paperMemo, setPaperMemo] = useState("");
  const [paperMemoStatus, setPaperMemoStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [paperMemoNotice, setPaperMemoNotice] = useState("");
  const memoSocketRef = useRef<WebSocket | null>(null);
  const paperMemoDirtyRef = useRef(false);
  const paperMemoRemoteValueRef = useRef("");

  useEffect(() => {
    setPaperMemoStatus("idle");
    setPaperMemoNotice("");
    memoSocketRef.current?.close();
    memoSocketRef.current = null;
    if (!currentPaperId) {
      setPaperMemo("");
      paperMemoDirtyRef.current = false;
      paperMemoRemoteValueRef.current = "";
      return;
    }
    setPaperMemo("");
    paperMemoDirtyRef.current = false;
    paperMemoRemoteValueRef.current = "";
    let cancelled = false;
    const loadMemo = async () => {
      try {
        const response = await getPaperMemo(currentPaperId);
        if (cancelled) {
          return;
        }
        const nextMemo = normalizeMemoText(response.memo);
        paperMemoRemoteValueRef.current = nextMemo;
        paperMemoDirtyRef.current = false;
        setPaperMemo(nextMemo);
      } catch {
        if (cancelled) {
          return;
        }
        paperMemoRemoteValueRef.current = "";
        paperMemoDirtyRef.current = false;
      }
    };
    void (async () => {
      await loadMemo();
      if (cancelled) {
        return;
      }
      const memoSocket = new WebSocket(toWebSocketUrl(`/papers/${encodeURIComponent(currentPaperId)}/memo/ws`));
      memoSocketRef.current = memoSocket;
      memoSocket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as { memo?: string };
          const nextMemo = normalizeMemoText(payload.memo);
          if (!paperMemoDirtyRef.current && nextMemo === paperMemoRemoteValueRef.current) {
            return;
          }
          paperMemoRemoteValueRef.current = nextMemo;
          paperMemoDirtyRef.current = false;
          setPaperMemo(nextMemo);
          setPaperMemoStatus("saved");
          setPaperMemoNotice("更新されました");
        } catch {
          // initial snapshot と更新通知だけを受ける
        }
      };
      memoSocket.onerror = () => {
        // メモは補助機能なので、接続失敗は画面全体に広げない
      };
    })();
    return () => {
      cancelled = true;
      const memoSocket = memoSocketRef.current;
      if (memoSocket) {
        memoSocket.close();
        if (memoSocketRef.current === memoSocket) {
          memoSocketRef.current = null;
        }
      }
    };
  }, [currentPaperId]);

  useEffect(() => {
    const paperId = currentPaperId;
    if (!paperId) {
      return;
    }
    if (
      !shouldSaveMemo({
        currentPaperId: paperId,
        isDirty: paperMemoDirtyRef.current,
        memo: paperMemo,
        remoteValue: paperMemoRemoteValueRef.current,
      })
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      setPaperMemoStatus("saving");
      void savePaperMemo(paperId, paperMemo)
        .then((response) => {
          const nextMemo = normalizeMemoText(response.memo);
          paperMemoDirtyRef.current = false;
          paperMemoRemoteValueRef.current = nextMemo;
          setPaperMemo(nextMemo);
          setPaperMemoStatus("saved");
          setPaperMemoNotice("保存されました");
        })
        .catch((caught: unknown) => {
          paperMemoDirtyRef.current = true;
          setPaperMemoStatus("error");
          setPaperMemoNotice("");
          if (onError) {
            onError(caught instanceof Error ? caught.message : "memo save failed");
          }
        });
    }, 300);
    return () => {
      window.clearTimeout(timer);
    };
  }, [paperMemo, currentPaperId]);

  useEffect(() => {
    if (paperMemoStatus !== "saved") {
      return;
    }
    const timer = window.setTimeout(() => {
      setPaperMemoStatus("idle");
    }, 1500);
    return () => {
      window.clearTimeout(timer);
    };
  }, [paperMemoStatus]);

  useEffect(() => {
    if (!paperMemoNotice) {
      return;
    }
    const timer = window.setTimeout(() => {
      setPaperMemoNotice("");
    }, 1500);
    return () => {
      window.clearTimeout(timer);
    };
  }, [paperMemoNotice]);

  const setMemo = (nextMemo: string) => {
    paperMemoDirtyRef.current = true;
    setPaperMemo(nextMemo);
  };

  return {
    paperMemo,
    setPaperMemo: setMemo,
    paperMemoStatus,
    paperMemoNotice,
    paperMemoRemoteValueRef,
    paperMemoDirtyRef,
  };
}
