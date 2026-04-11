export function canSendSessionAction(params: { currentSessionId: string | null; wsConnected: boolean }): boolean {
  return Boolean(params.currentSessionId && params.wsConnected);
}

export function shouldHighlightNextCandidateAction(nextCandidatePaperId: string | null): boolean {
  return nextCandidatePaperId !== null;
}
