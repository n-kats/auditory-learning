export function resolveDisplayedNextCandidatePaperId(params: {
  selectedNextCandidatePaperId: string | null;
  nextPaperId: string | null;
}): string | null {
  return params.selectedNextCandidatePaperId ?? params.nextPaperId;
}
