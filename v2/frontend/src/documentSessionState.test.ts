import {
  applyDocumentSessionFlowEvent,
  createDocumentSessionFlowState,
  simulateDocumentSessionFlow,
} from "./documentSessionState";

describe("documentSessionState", () => {
  it("simulates start, page load, websocket and favorite changes", () => {
    const state = simulateDocumentSessionFlow([
      { type: "start_requested", draft_url: "https://arxiv.org/pdf/2604.16347" },
      { type: "start_succeeded", request_id: "session-1", source_url: "https://arxiv.org/pdf/2604.16347", page_num: 12 },
      {
        type: "page_load_started",
        request_id: "session-1",
        page: 1,
        regenerate: false,
        load_id: 1,
      },
      {
        type: "page_load_succeeded",
        request_id: "session-1",
        page: 1,
        load_id: 1,
        explanation: "hello",
        image_url: "blob:image",
        audio_url: "blob:audio",
        audio_status: "ready",
      },
      {
        type: "ws_event",
        event: {
          type: "page_updated",
          request_id: "session-1",
          current_page: 4,
        },
      },
      {
        type: "favorite_toggled",
        request_id: "session-1",
        is_favorited: true,
      },
    ]);

    expect(state).toEqual({
      draftUrl: "https://arxiv.org/pdf/2604.16347",
      sourceUrl: "https://arxiv.org/pdf/2604.16347",
      requestId: "session-1",
      maxPage: 12,
      currentPage: 4,
      totalGenerationCount: 0,
      totalGenerationElapsedMs: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalCostUsd: 0,
      jumpPageValue: "4",
      explanation: "hello",
      imageUrl: "blob:image",
      audioUrl: "blob:audio",
      error: null,
      audioStatusText: "音声:ok",
      audioStatusError: null,
      generationStatusText: "",
      isInitializing: false,
      isLoadingPage: false,
      isRegenerating: false,
      isFavorited: true,
      activeLoadId: null,
    });
  });

  it("keeps no-op websocket events from changing state", () => {
    const state = createDocumentSessionFlowState({
      requestId: "session-1",
      currentPage: 3,
      maxPage: 10,
      isFavorited: false,
    });

    const nextState = applyDocumentSessionFlowEvent(state, {
      type: "ws_event",
      event: {
        type: "page_updated",
        request_id: "session-2",
        current_page: 9,
      },
    });

    expect(nextState).toBe(state);
  });

  it("ignores stale page load results after a newer load starts", () => {
    const state = simulateDocumentSessionFlow([
      { type: "start_requested", draft_url: "https://arxiv.org/pdf/2604.16347" },
      { type: "start_succeeded", request_id: "session-1", source_url: "https://arxiv.org/pdf/2604.16347", page_num: 12 },
      { type: "page_load_started", request_id: "session-1", page: 1, regenerate: false, load_id: 1 },
      { type: "page_load_started", request_id: "session-1", page: 2, regenerate: false, load_id: 2 },
      {
        type: "page_load_succeeded",
        request_id: "session-1",
        page: 1,
        load_id: 1,
        explanation: "stale",
        image_url: "blob:old-image",
        audio_url: "blob:old-audio",
        audio_status: "ready",
      },
      {
        type: "page_load_succeeded",
        request_id: "session-1",
        page: 2,
        load_id: 2,
        explanation: "fresh",
        image_url: "blob:new-image",
        audio_url: "blob:new-audio",
        audio_status: "ready",
      },
    ]);

    expect(state.currentPage).toBe(2);
    expect(state.explanation).toBe("fresh");
    expect(state.imageUrl).toBe("blob:new-image");
    expect(state.audioUrl).toBe("blob:new-audio");
    expect(state.activeLoadId).toBeNull();
  });

  it("restores a session snapshot for continue from saved state", () => {
    const state = simulateDocumentSessionFlow([
      { type: "resume_requested", request_id: "session-1" },
      {
        type: "resume_succeeded",
        snapshot: {
          request_id: "session-1",
          source_url: "https://arxiv.org/pdf/2604.16347",
          page_num: 12,
          current_page: 4,
          is_favorited: true,
          total_generation_count: 7,
          total_generation_elapsed_ms: 2345,
          total_input_tokens: 111,
          total_output_tokens: 222,
          total_cost_usd: 0.012345,
          created_at: "2026-05-03T00:00:00Z",
          updated_at: "2026-05-03T00:00:00Z",
        },
      },
    ]);

    expect(state.draftUrl).toBe("https://arxiv.org/pdf/2604.16347");
    expect(state.sourceUrl).toBe("https://arxiv.org/pdf/2604.16347");
    expect(state.requestId).toBe("session-1");
    expect(state.currentPage).toBe(4);
    expect(state.maxPage).toBe(12);
    expect(state.jumpPageValue).toBe("4");
    expect(state.isFavorited).toBe(true);
    expect(state.totalGenerationCount).toBe(7);
    expect(state.totalGenerationElapsedMs).toBe(2345);
    expect(state.totalInputTokens).toBe(111);
    expect(state.totalOutputTokens).toBe(222);
    expect(state.totalCostUsd).toBe(0.012345);
  });

  it("keeps favorite state in sync with websocket and local toggle", () => {
    const state = simulateDocumentSessionFlow([
      { type: "start_requested", draft_url: "https://arxiv.org/pdf/2604.16347" },
      { type: "start_succeeded", request_id: "session-1", source_url: "https://arxiv.org/pdf/2604.16347", page_num: 12 },
      { type: "favorite_toggled", request_id: "session-1", is_favorited: true },
      {
        type: "ws_event",
        event: {
          type: "favorite_toggled",
          request_id: "session-1",
          is_favorited: false,
        },
      },
    ]);

    expect(state.isFavorited).toBe(false);
  });

  it("marks failed page loads and clears the spinner state", () => {
    const state = applyDocumentSessionFlowEvent(
      createDocumentSessionFlowState({ requestId: "session-1", currentPage: 2, isLoadingPage: true, activeLoadId: 1 }),
      {
        type: "page_load_failed",
        request_id: "session-1",
        page: 2,
        load_id: 1,
        error: "boom",
      },
    );

    expect(state.error).toBe("boom");
    expect(state.audioStatusText).toBe("");
    expect(state.isLoadingPage).toBe(false);
    expect(state.isRegenerating).toBe(false);
  });

  it("clears previous page content when a new page load starts", () => {
    const state = applyDocumentSessionFlowEvent(
      createDocumentSessionFlowState({
        requestId: "session-1",
        currentPage: 3,
        explanation: "old explanation",
        imageUrl: "blob:old-image",
        audioUrl: "blob:old-audio",
      }),
      {
        type: "page_load_started",
        request_id: "session-1",
        page: 4,
        regenerate: false,
        load_id: 2,
      },
    );

    expect(state.currentPage).toBe(4);
    expect(state.explanation).toBe("");
    expect(state.imageUrl).toBeNull();
    expect(state.audioUrl).toBeNull();
    expect(state.audioStatusText).toBe("音声:確認中");
    expect(state.generationStatusText).toBe("");
    expect(state.isLoadingPage).toBe(true);
    expect(state.activeLoadId).toBe(2);
  });

  it("streams page image and explanation before the final page load succeeds", () => {
    const stateAfterImage = applyDocumentSessionFlowEvent(createDocumentSessionFlowState({ requestId: "session-1", activeLoadId: 1 }), {
      type: "page_image_loaded",
      request_id: "session-1",
      page: 2,
      load_id: 1,
      image_url: "blob:image",
    });
    const stateAfterExplanation = applyDocumentSessionFlowEvent(stateAfterImage, {
      type: "page_explanation_loaded",
      request_id: "session-1",
      page: 2,
      load_id: 1,
      explanation: "streamed explanation",
      audio_status: "ready",
    });

    expect(stateAfterImage.imageUrl).toBe("blob:image");
    expect(stateAfterImage.explanation).toBe("");
    expect(stateAfterExplanation.imageUrl).toBe("blob:image");
    expect(stateAfterExplanation.explanation).toBe("streamed explanation");
    expect(stateAfterExplanation.audioStatusText).toBe("音声:ok");
  });

  it("tracks generation status messages", () => {
    const started = applyDocumentSessionFlowEvent(createDocumentSessionFlowState({ requestId: "session-1" }), {
      type: "generation_started",
      request_id: "session-1",
      page: 7,
    });
    const finished = applyDocumentSessionFlowEvent(started, {
      type: "generation_finished",
      request_id: "session-1",
      page: 7,
    });

    expect(started.generationStatusText).toBe("7ページ目生成中");
    expect(finished.generationStatusText).toBe("");
  });
});
