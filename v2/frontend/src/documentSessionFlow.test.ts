import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("./api", () => {
  return {
    fetchExplanation: vi.fn(),
    fetchPageAudio: vi.fn(),
    fetchPageImage: vi.fn(),
    initDocument: vi.fn(),
    isNetworkFetchError: vi.fn(),
    regenerateExplanation: vi.fn(),
  };
});

import { fetchExplanation, fetchPageAudio, fetchPageImage, regenerateExplanation } from "./api";
import { loadDocumentPage } from "./documentSessionFlow";
import type { DocumentSessionFlowEvent } from "./documentSessionState";

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

class FakeObjectUrlStore {
  private values = new Map<string, string>();

  get(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  set(key: string): string {
    const value = `blob:${key}`;
    this.values.set(key, value);
    return value;
  }

  delete(key: string): void {
    this.values.delete(key);
  }

  clear(): void {
    this.values.clear();
  }
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("loadDocumentPage", () => {
  it("streams image and explanation before the final success event", async () => {
    const image = deferred<Blob>();
    const explanation = deferred<{
      explanation: string;
      speech_text: string;
      audio_status?: "ready" | "failed";
      audio_error?: string | null;
    }>();
    const audio = deferred<Blob>();
    const events: DocumentSessionFlowEvent[] = [];
    const imageStore = new FakeObjectUrlStore();
    const audioStore = new FakeObjectUrlStore();

    vi.mocked(fetchPageImage).mockReturnValueOnce(image.promise);
    vi.mocked(fetchExplanation).mockReturnValueOnce(explanation.promise);
    vi.mocked(fetchPageAudio).mockReturnValueOnce(audio.promise);
    vi.mocked(regenerateExplanation).mockRejectedValue(new Error("unexpected regenerate"));

    const loadPromise = loadDocumentPage({
      requestId: "session-1",
      page: 2,
      imageStore,
      audioStore,
      dispatchFlowEvent: (event) => {
        events.push(event);
      },
      sequenceRef: { current: 0 },
    });

    explanation.resolve({ explanation: "streamed explanation", speech_text: "streamed speech", audio_status: "ready" });
    await Promise.resolve();

    expect(events.some((event) => event.type === "page_explanation_loaded")).toBe(true);
    expect(events.some((event) => event.type === "page_load_succeeded")).toBe(false);

    image.resolve(new Blob(["image"]));
    await Promise.resolve();

    audio.resolve(new Blob(["audio"]));
    await loadPromise;

    const eventTypes = events.map((event) => event.type);
    expect(eventTypes).toContain("page_image_loaded");
    expect(eventTypes).toContain("page_explanation_loaded");
    expect(eventTypes).toContain("page_load_succeeded");
    expect(eventTypes.indexOf("page_image_loaded")).toBeLessThan(eventTypes.indexOf("page_load_succeeded"));
    expect(eventTypes.indexOf("page_explanation_loaded")).toBeLessThan(eventTypes.indexOf("page_load_succeeded"));
  });
});
