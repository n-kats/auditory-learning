export function getStartSessionError(params: { databaseReady: boolean; sourceUrl: string }): string | null {
  if (!params.databaseReady) {
    return "データベース初期化中です。しばらくしてから再試行してください。";
  }
  if (params.sourceUrl.trim().length === 0) {
    return "開始URLを入力してください。";
  }
  return null;
}

export function getResumeSessionError(params: { databaseReady: boolean; sessionId: string }): string | null {
  if (!params.databaseReady) {
    return "データベース初期化中です。しばらくしてから再試行してください。";
  }
  if (params.sessionId.trim().length === 0) {
    return "session_id がありません。";
  }
  return null;
}

export function getNextSessionError(params: { databaseReady: boolean; hasOpenSession: boolean }): string | null {
  if (!params.databaseReady) {
    return "データベース初期化中です。しばらくしてから再試行してください。";
  }
  if (!params.hasOpenSession) {
    return "次へ進める session がありません。";
  }
  return null;
}

export function getRegenerateSessionError(params: { databaseReady: boolean; hasOpenSession: boolean }): string | null {
  if (!params.databaseReady) {
    return "データベース初期化中です。しばらくしてから再試行してください。";
  }
  if (!params.hasOpenSession) {
    return "再生成する session がありません。";
  }
  return null;
}

export function getResumeAudioError(params: {
  loading: boolean;
  currentSessionId: string | null;
  audioUrlsLength: number;
}): string | null {
  if (params.loading) {
    return "処理中です。しばらくしてから再開してください。";
  }
  if (!params.currentSessionId || params.audioUrlsLength === 0) {
    return "再開する再生がありません。";
  }
  return null;
}
