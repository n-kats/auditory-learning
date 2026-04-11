import { describe, expect, it } from "vitest";
import {
  getNextSessionError,
  getRegenerateSessionError,
  getResumeAudioError,
  getResumeSessionError,
  getStartSessionError,
} from "./sessionOperationChecks";

describe("sessionOperationChecks", () => {
  it("validates start input", () => {
    expect(getStartSessionError({ databaseReady: false, sourceUrl: "https://arxiv.org/abs/1" })).toContain("データベース");
    expect(getStartSessionError({ databaseReady: true, sourceUrl: "" })).toBe("開始URLを入力してください。");
    expect(getStartSessionError({ databaseReady: true, sourceUrl: "https://arxiv.org/abs/1" })).toBeNull();
  });

  it("validates resume input", () => {
    expect(getResumeSessionError({ databaseReady: false, sessionId: "s" })).toContain("データベース");
    expect(getResumeSessionError({ databaseReady: true, sessionId: "" })).toBe("session_id がありません。");
    expect(getResumeSessionError({ databaseReady: true, sessionId: "s" })).toBeNull();
  });

  it("validates next and regenerate input", () => {
    expect(getNextSessionError({ databaseReady: false, hasOpenSession: true })).toContain("データベース");
    expect(getNextSessionError({ databaseReady: true, hasOpenSession: false })).toBe("次へ進める session がありません。");
    expect(getRegenerateSessionError({ databaseReady: false, hasOpenSession: true })).toContain("データベース");
    expect(getRegenerateSessionError({ databaseReady: true, hasOpenSession: false })).toBe("再生成する session がありません。");
  });

  it("validates resume audio", () => {
    expect(getResumeAudioError({ loading: true, currentSessionId: "s", audioUrlsLength: 1 })).toBe("処理中です。しばらくしてから再開してください。");
    expect(getResumeAudioError({ loading: false, currentSessionId: null, audioUrlsLength: 1 })).toBe("再開する再生がありません。");
    expect(getResumeAudioError({ loading: false, currentSessionId: "s", audioUrlsLength: 0 })).toBe("再開する再生がありません。");
    expect(getResumeAudioError({ loading: false, currentSessionId: "s", audioUrlsLength: 1 })).toBeNull();
  });
});
