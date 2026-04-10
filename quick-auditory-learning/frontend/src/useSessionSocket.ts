import { useEffect, useRef, useState } from "react";

import { toWebSocketUrl, type SessionEventMessage } from "./api";
import { buildResumePayload, buildStopPayload, shouldReconnectAfterClose, type SessionSocketStatus } from "./sessionSocket";

type UseSessionSocketOptions = {
  onError: (message: string) => void;
};

export function useSessionSocket({ onError }: UseSessionSocketOptions) {
  const [wsStatus, setWsStatus] = useState<SessionSocketStatus>("idle");
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const lastEventSeqRef = useRef(0);
  const manualStopRef = useRef(false);
  const onMessageRef = useRef<(message: SessionEventMessage) => void | Promise<void>>(() => undefined);

  const scheduleReconnect = () => {
    const sessionId = sessionIdRef.current;
    if (sessionId === null) {
      return;
    }
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
    }
    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      openSessionSocket(buildResumePayload(sessionId, lastEventSeqRef.current), true);
    }, 1000);
    setWsStatus("reconnecting");
  };

  const closeSocket = (sendStop: boolean) => {
    const socket = socketRef.current;
    if (sendStop && socket && socket.readyState === WebSocket.OPEN && sessionIdRef.current) {
      socket.send(JSON.stringify(buildStopPayload(sessionIdRef.current)));
    }
    manualStopRef.current = true;
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    socket?.close();
    socketRef.current = null;
    setWsStatus("closed");
  };

  const openSessionSocket = (initialMessage?: Record<string, unknown>, isReconnect = false) => {
    manualStopRef.current = false;
    const socket = new WebSocket(toWebSocketUrl("/sessions/ws"));
    socketRef.current = socket;
    setWsStatus(isReconnect ? "reconnecting" : "connecting");
    socket.onopen = () => {
      if (socketRef.current !== socket) {
        return;
      }
      setWsStatus("connected");
      if (initialMessage) {
        socket.send(JSON.stringify(initialMessage));
      }
    };
    socket.onmessage = (event) => {
      if (socketRef.current !== socket) {
        return;
      }
      try {
        const message = JSON.parse(event.data) as SessionEventMessage;
        void onMessageRef.current(message);
      } catch (caught: unknown) {
        onError(caught instanceof Error ? caught.message : "invalid session message");
      }
    };
    socket.onerror = () => {
      if (socketRef.current !== socket) {
        return;
      }
      setWsStatus("reconnecting");
    };
    socket.onclose = () => {
      if (socketRef.current !== socket) {
        return;
      }
      socketRef.current = null;
      if (shouldReconnectAfterClose(manualStopRef.current, sessionIdRef.current)) {
        scheduleReconnect();
      } else {
        setWsStatus("closed");
      }
    };
  };

  useEffect(() => {
    return () => {
      manualStopRef.current = true;
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      socketRef.current?.close();
    };
  }, []);

  return {
    socketRef,
    reconnectTimerRef,
    sessionIdRef,
    lastEventSeqRef,
    manualStopRef,
    setManualStop: (value: boolean) => {
      manualStopRef.current = value;
    },
    setSessionId: (sessionId: string | null) => {
      sessionIdRef.current = sessionId;
    },
    setLastEventSeq: (seq: number) => {
      lastEventSeqRef.current = seq;
    },
    getSessionId: () => sessionIdRef.current,
    getLastEventSeq: () => lastEventSeqRef.current,
    wsStatus,
    setWsStatus,
    setMessageHandler: (handler: (message: SessionEventMessage) => void | Promise<void>) => {
      onMessageRef.current = handler;
    },
    openSessionSocket,
    closeSocket,
  };
}
