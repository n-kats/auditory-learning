const DEFAULT_PREVIEW_ZOOM = 1;
const MIN_PREVIEW_ZOOM = 0.5;
const MAX_PREVIEW_ZOOM = 3;
const PREVIEW_ZOOM_STEP = 0.15;

export function clampPreviewZoom(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_PREVIEW_ZOOM;
  }
  return Math.min(Math.max(value, MIN_PREVIEW_ZOOM), MAX_PREVIEW_ZOOM);
}

export function adjustPreviewZoomForWheel(currentZoom: number, deltaY: number): number {
  if (!Number.isFinite(currentZoom)) {
    return DEFAULT_PREVIEW_ZOOM;
  }
  if (!Number.isFinite(deltaY) || deltaY === 0) {
    return clampPreviewZoom(currentZoom);
  }

  const nextZoom = currentZoom + (deltaY > 0 ? -PREVIEW_ZOOM_STEP : PREVIEW_ZOOM_STEP);
  return clampPreviewZoom(nextZoom);
}
