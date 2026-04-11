export type SessionOperationKind = "start" | "resume" | "next" | "regenerate";

export type SessionOperationPatch = {
  error: string | null;
  loading: boolean;
  pendingAction: SessionOperationKind | "idle";
  backendNotices: string[];
  shouldAutoPlay?: boolean;
};

export function resolveShouldAutoPlayOnAdvance(isPlaying: boolean, shouldAutoPlay: boolean): boolean {
  return isPlaying || shouldAutoPlay;
}

export function buildSessionOperationStartPatch(
  kind: SessionOperationKind,
  options?: { shouldAutoPlay?: boolean },
): SessionOperationPatch {
  return {
    error: null,
    loading: true,
    pendingAction: kind,
    backendNotices: [],
    shouldAutoPlay: options?.shouldAutoPlay ?? true,
  };
}

export function buildSessionOperationIdlePatch(): SessionOperationPatch {
  return {
    error: null,
    loading: false,
    pendingAction: "idle",
    backendNotices: [],
  };
}

export function buildSessionOperationFailurePatch(
  message: string,
): SessionOperationPatch {
  return {
    error: message,
    loading: false,
    pendingAction: "idle",
    backendNotices: [],
  };
}
