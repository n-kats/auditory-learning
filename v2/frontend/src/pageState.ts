export function clampPage(page: number, maxPage: number): number {
  if (!Number.isFinite(page) || maxPage <= 0) {
    return 1;
  }
  return Math.min(Math.max(Math.trunc(page), 1), maxPage);
}

export function parseJumpPage(value: string, maxPage: number): number | null {
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return null;
  }
  const numeric = Number(trimmed);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  return clampPage(numeric, maxPage);
}

export function formatPageLabel(currentPage: number, maxPage: number): string {
  return `${currentPage} / ${maxPage}`;
}
