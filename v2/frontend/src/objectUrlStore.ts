export class ObjectUrlStore {
  private urls = new Map<string, string>();

  get(key: string): string | null {
    return this.urls.get(key) ?? null;
  }

  set(key: string, blob: Blob): string {
    const previous = this.urls.get(key);
    if (previous) {
      URL.revokeObjectURL(previous);
    }
    const next = URL.createObjectURL(blob);
    this.urls.set(key, next);
    return next;
  }

  delete(key: string): void {
    const current = this.urls.get(key);
    if (!current) {
      return;
    }
    URL.revokeObjectURL(current);
    this.urls.delete(key);
  }

  clear(): void {
    for (const current of this.urls.values()) {
      URL.revokeObjectURL(current);
    }
    this.urls.clear();
  }
}
