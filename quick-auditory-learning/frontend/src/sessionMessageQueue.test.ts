import { describe, expect, it, vi } from "vitest";
import { createSequentialAsyncQueue } from "./sessionMessageQueue";

describe("sessionMessageQueue", () => {
  it("processes tasks in received order even when earlier tasks await longer", async () => {
    const onError = vi.fn();
    const queue = createSequentialAsyncQueue(onError);
    const events: string[] = [];
    let resolveFirst: (() => void) | null = null;

    const first = queue.enqueue(
      () =>
        new Promise<void>((resolve) => {
          events.push("first:start");
          resolveFirst = () => {
            events.push("first:end");
            resolve();
          };
        }),
    );
    const second = queue.enqueue(async () => {
      events.push("second:start");
      events.push("second:end");
    });

    await Promise.resolve();
    expect(events).toEqual(["first:start"]);
    expect(resolveFirst).not.toBeNull();

    const firstResolver = resolveFirst;
    if (firstResolver === null) {
      throw new Error("expected first resolver");
    }
    (firstResolver as () => void)();
    await first;
    await second;

    expect(events).toEqual(["first:start", "first:end", "second:start", "second:end"]);
    expect(onError).not.toHaveBeenCalled();
  });

  it("continues after task failure", async () => {
    const onError = vi.fn();
    const queue = createSequentialAsyncQueue(onError);
    const events: string[] = [];

    await queue.enqueue(async () => {
      events.push("first");
      throw new Error("boom");
    });
    await queue.enqueue(async () => {
      events.push("second");
    });

    expect(events).toEqual(["first", "second"]);
    expect(onError).toHaveBeenCalledTimes(1);
  });
});
