export function normalizeMemoText(value: string | null | undefined): string {
  return value ?? "";
}

export function shouldSaveMemo(params: {
  currentPaperId: string | null;
  isDirty: boolean;
  memo: string;
  remoteValue: string;
}): boolean {
  return Boolean(params.currentPaperId && params.isDirty && params.memo !== params.remoteValue);
}
