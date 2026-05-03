import { applyDocumentSessionSyncEvent, createDocumentSessionSyncState } from "./documentSessionSync";

describe("documentSessionSync", () => {
  it("applies snapshot, page and favorite updates for the current session", () => {
    const state = applyDocumentSessionSyncEvent(
      createDocumentSessionSyncState({ requestId: "session-1", currentPage: 1, maxPage: 8, isFavorited: false }),
      {
        type: "session_snapshot",
        request_id: "session-1",
        page_num: 12,
        current_page: 3,
        is_favorited: true,
        total_generation_count: 5,
        total_generation_elapsed_ms: 1200,
        total_input_tokens: 200,
        total_output_tokens: 400,
        total_cost_usd: 0.123456,
      },
    );

    const nextState = applyDocumentSessionSyncEvent(state, {
      type: "page_updated",
      request_id: "session-1",
      current_page: 7,
    });

    const finalState = applyDocumentSessionSyncEvent(nextState, {
      type: "favorite_toggled",
      request_id: "session-1",
      is_favorited: false,
    });

    expect(finalState).toEqual({
      requestId: "session-1",
      currentPage: 7,
      maxPage: 12,
      isFavorited: false,
      totalGenerationCount: 5,
      totalGenerationElapsedMs: 1200,
      totalInputTokens: 200,
      totalOutputTokens: 400,
      totalCostUsd: 0.123456,
    });
  });

  it("ignores events for another session", () => {
    const state = createDocumentSessionSyncState({
      requestId: "session-1",
      currentPage: 2,
      maxPage: 10,
      isFavorited: true,
    });

    const nextState = applyDocumentSessionSyncEvent(state, {
      type: "page_updated",
      request_id: "session-2",
      current_page: 9,
    });

    expect(nextState).toBe(state);
  });

  it("returns the same state for a no-op snapshot", () => {
    const state = createDocumentSessionSyncState({
      requestId: "session-1",
      currentPage: 3,
      maxPage: 12,
      isFavorited: false,
      totalGenerationCount: 0,
      totalGenerationElapsedMs: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalCostUsd: 0,
    });

    const nextState = applyDocumentSessionSyncEvent(state, {
      type: "session_snapshot",
      request_id: "session-1",
      current_page: 3,
      page_num: 12,
      is_favorited: false,
      total_generation_count: 0,
      total_generation_elapsed_ms: 0,
      total_input_tokens: 0,
      total_output_tokens: 0,
      total_cost_usd: 0,
    });

    expect(nextState).toBe(state);
  });
});
