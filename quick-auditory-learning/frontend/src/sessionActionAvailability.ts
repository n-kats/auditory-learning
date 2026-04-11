export function canSendSessionAction(params: { currentSessionId: string | null; wsConnected: boolean }): boolean {
  return Boolean(params.currentSessionId && params.wsConnected);
}
