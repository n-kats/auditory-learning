export type StartSessionCommandInput = {
  sourceUrl: string;
  modelName: string;
  includeOldVectors: boolean;
  limit: number;
  route1Weight: number;
  route2Weight: number;
  seed: number | null;
  searchModes: string[];
};

export function buildStartSessionCommand(input: StartSessionCommandInput): Record<string, unknown> {
  return {
    type: "start",
    source_url: input.sourceUrl,
    model_name: input.modelName,
    include_old_vectors: input.includeOldVectors,
    limit: input.limit,
    route1_weight: input.route1Weight,
    route2_weight: input.route2Weight,
    seed: input.seed,
    search_modes: input.searchModes,
  };
}

export function buildStopSessionCommand(sessionId: string): Record<string, unknown> {
  return {
    type: "stop",
    session_id: sessionId,
  };
}

export function buildNextSessionCommand(sessionId: string): Record<string, unknown> {
  return {
    type: "next",
    session_id: sessionId,
  };
}

export function buildRegenerateSessionCommand(sessionId: string): Record<string, unknown> {
  return {
    type: "regenerate",
    session_id: sessionId,
  };
}

export function buildPlaybackStartedSessionCommand(sessionId: string, paperId: string): Record<string, unknown> {
  return {
    type: "playback_started",
    session_id: sessionId,
    paper_id: paperId,
  };
}

export function buildSetNextCandidateCommand(sessionId: string, paperId: string): Record<string, unknown> {
  return {
    type: "set_next_candidate",
    session_id: sessionId,
    paper_id: paperId,
  };
}
