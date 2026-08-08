import { useEffect, useState } from "react";

import type { SessionSummary } from "./api";
import { useDocumentSession } from "./hooks/useDocumentSession";
import { useSessionDirectory } from "./hooks/useSessionDirectory";
import { FavoritesPage } from "./pages/FavoritesPage";
import { SessionPage } from "./pages/SessionPage";
import { StartPage } from "./pages/StartPage";

type AppPage = "start" | "favorites" | "session";

function getPageFromPathname(): AppPage {
  if (window.location.pathname.startsWith("/favorites")) {
    return "favorites";
  }
  if (window.location.pathname.startsWith("/session")) {
    return "session";
  }
  return "start";
}

export default function App() {
  const session = useDocumentSession();
  const directory = useSessionDirectory();
  const [page, setPage] = useState<AppPage>(() => (typeof window === "undefined" ? "start" : getPageFromPathname()));

  const isSessionActive = session.requestId !== null;

  useEffect(() => {
    const handlePopState = () => {
      setPage(getPageFromPathname());
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const navigateTo = (nextPage: AppPage) => {
    if (typeof window !== "undefined") {
      const nextPath = nextPage === "session" ? "/session" : nextPage === "favorites" ? "/favorites" : "/";
      if (window.location.pathname !== nextPath) {
        window.history.pushState({}, "", nextPath);
      }
    }
    setPage(nextPage);
  };

  const handleGoToStart = () => {
    navigateTo("start");
  };

  const handleGoToFavorites = () => {
    navigateTo("favorites");
  };

  const handleGoToSession = () => {
    if (!isSessionActive) {
      return;
    }
    navigateTo("session");
  };

  const handleStartDocument = async (event: Parameters<typeof session.startDocument>[0]) => {
    await session.startDocument(event);
    navigateTo("session");
  };

  const handleUploadDocument = async (file: File) => {
    await session.startDocumentFromUpload(file);
    navigateTo("session");
  };

  const handleResumeDocument = async (snapshot: SessionSummary) => {
    await session.resumeDocumentByRequestId(snapshot.request_id);
    navigateTo("session");
  };

  return (
    <div className="app-shell">
      {isSessionActive ? <audio ref={session.audioRef} src={session.audioUrl ?? undefined} preload="auto" /> : null}

      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />

      <div className="app-frame">
        <header className="app-topbar">
          <div className="app-brand-row">
            <p className="app-title">AUDITORY LEARNING V2</p>
            <span className="app-subtitle">db: ok</span>
            {session.audioStatusText ? (
              <span
                className={`app-subtitle app-audio-status${session.audioStatusText.includes("失敗") ? " is-failed" : session.audioStatusText.includes("確認") ? " is-busy" : " is-idle"}`}
                title={session.audioStatusError ?? undefined}
              >
                {session.audioStatusText}
              </span>
            ) : null}
            {session.generationStatusText ? (
              <span className="app-subtitle app-generation-status" title={session.generationStatusText}>
                {session.generationStatusText}
              </span>
            ) : null}
          </div>
          <div className="topbar-links" aria-label="navigation">
            <button type="button" className={`topbar-link-button${page === "start" ? " is-active" : ""}`} onClick={handleGoToStart}>
              開始・続きから
            </button>
            <button type="button" className={`topbar-link-button${page === "favorites" ? " is-active" : ""}`} onClick={handleGoToFavorites}>
              お気に入り
            </button>
            {isSessionActive ? (
              <button type="button" className={`topbar-link-button${page === "session" ? " is-active" : ""}`} onClick={handleGoToSession}>
                現在のセッション
              </button>
            ) : null}
          </div>
        </header>

        {session.error ? <section className="card error-card">{session.error}</section> : null}

        {page === "session" && isSessionActive ? (
          <SessionPage session={session} />
        ) : page === "favorites" ? (
          <FavoritesPage />
        ) : (
          <StartPage
            draftUrl={session.draftUrl}
            currentSessionId={session.requestId}
            draftExplainPromptText={session.draftExplainPromptText}
            draftSpeakPromptText={session.draftSpeakPromptText}
            modelName={session.draftModelName}
            reasoningEffort={session.draftReasoningEffort}
            sessions={directory.sessions}
            isInitializing={session.isInitializing}
            isLoadingSessions={directory.isLoading}
            sessionsError={directory.error}
            onContinue={(snapshot) => {
              void handleResumeDocument(snapshot);
            }}
            onDraftUrlChange={session.setDraftUrl}
            onDraftExplainPromptTextChange={session.setDraftExplainPromptText}
            onDraftSpeakPromptTextChange={session.setDraftSpeakPromptText}
            onModelNameChange={session.setDraftModelName}
            onReasoningEffortChange={session.setDraftReasoningEffort}
            onSubmit={(event) => {
              void handleStartDocument(event);
            }}
            onUpload={(file) => {
              void handleUploadDocument(file);
            }}
          />
        )}
      </div>
    </div>
  );
}
