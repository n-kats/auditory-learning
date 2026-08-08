import type { FormEvent } from "react";

import type { SessionSummary } from "../api";
import { SessionDirectory } from "../components/SessionDirectory";

type StartPageProps = {
  draftUrl: string;
  draftExplainPromptText: string;
  draftSpeakPromptText: string;
  modelName: string;
  reasoningEffort: string;
  sessions: SessionSummary[];
  currentSessionId: string | null;
  isInitializing: boolean;
  isLoadingSessions: boolean;
  sessionsError: string | null;
  onContinue: (session: SessionSummary) => void;
  onDraftUrlChange: (value: string) => void;
  onDraftExplainPromptTextChange: (value: string) => void;
  onDraftSpeakPromptTextChange: (value: string) => void;
  onModelNameChange: (value: string) => void;
  onReasoningEffortChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onUpload: (file: File) => void;
};

export function StartPage(props: StartPageProps) {
  return <SessionDirectory {...props} />;
}
