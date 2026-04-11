export type SequentialAsyncQueue = {
  enqueue: (task: () => void | Promise<void>) => Promise<void>;
};

export function createSequentialAsyncQueue(onError: (error: unknown) => void): SequentialAsyncQueue {
  let tail: Promise<void> = Promise.resolve();

  return {
    enqueue(task) {
      tail = tail
        .then(async () => {
          await task();
        })
        .catch((caught: unknown) => {
          onError(caught);
        })
        .then(() => undefined);
      return tail;
    },
  };
}
