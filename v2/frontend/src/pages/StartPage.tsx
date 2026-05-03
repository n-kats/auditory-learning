import type { FormEvent } from "react";

import type { SessionSummary } from "../api";
import { SessionDirectory } from "../components/SessionDirectory";

type StartPageProps = {
  draftUrl: string;
  draftExplainPromptText: string;
  draftSpeekPromptText: string;
  modelName: string;
  sessions: SessionSummary[];
  currentSessionId: string | null;
  isInitializing: boolean;
  isLoadingSessions: boolean;
  sessionsError: string | null;
  onContinue: (session: SessionSummary) => void;
  onDraftUrlChange: (value: string) => void;
  onDraftExplainPromptTextChange: (value: string) => void;
  onDraftSpeekPromptTextChange: (value: string) => void;
  onModelNameChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
};

export function StartPage(props: StartPageProps) {
  return <SessionDirectory {...props} />;
}
