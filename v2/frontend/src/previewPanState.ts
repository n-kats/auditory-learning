export type PreviewPan = {
  x: number;
  y: number;
};

export function applyPreviewDrag(origin: PreviewPan, deltaX: number, deltaY: number): PreviewPan {
  if (!Number.isFinite(origin.x) || !Number.isFinite(origin.y)) {
    return { x: 0, y: 0 };
  }
  if (!Number.isFinite(deltaX) || !Number.isFinite(deltaY)) {
    return origin;
  }
  return {
    x: origin.x + deltaX,
    y: origin.y + deltaY,
  };
}
