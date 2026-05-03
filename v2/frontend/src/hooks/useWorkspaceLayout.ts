import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject, type WheelEvent as ReactWheelEvent } from "react";

import { adjustPreviewZoomForWheel } from "../previewZoomState";
import { applyPreviewDrag } from "../previewPanState";

type UseWorkspaceLayoutResult = {
  isMobileWorkspace: boolean;
  workspaceGridRef: RefObject<HTMLElement | null>;
  workspaceGridColumns: string;
  workspaceSplit: number;
  previewZoom: number;
  previewPanX: number;
  previewPanY: number;
  isMainCollapsed: boolean;
  isPreviewCollapsed: boolean;
  setWorkspaceSplit: (value: number) => void;
  resetPreviewZoom: () => void;
  resetPreviewPan: () => void;
  onPreviewWheel: (event: ReactWheelEvent<HTMLDivElement>) => void;
  onPreviewPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
  onDividerPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
};

export function useWorkspaceLayout(): UseWorkspaceLayoutResult {
  const [isMobileWorkspace, setIsMobileWorkspace] = useState(() => (typeof window !== "undefined" ? window.innerWidth < 680 : false));
  const [workspaceSplit, setWorkspaceSplit] = useState(0.5);
  const [previewZoom, setPreviewZoom] = useState(1);
  const [previewPanX, setPreviewPanX] = useState(0);
  const [previewPanY, setPreviewPanY] = useState(0);
  const workspaceGridRef = useRef<HTMLElement | null>(null);
  const workspaceDraggingRef = useRef(false);
  const previewDraggingRef = useRef(false);
  const previewDragOriginRef = useRef({ x: 0, y: 0 });
  const previewDragStartRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const updateWorkspaceMode = () => {
      setIsMobileWorkspace(window.innerWidth < 680);
    };

    updateWorkspaceMode();
    window.addEventListener("resize", updateWorkspaceMode);
    return () => window.removeEventListener("resize", updateWorkspaceMode);
  }, []);

  const resetPreviewZoom = useCallback(() => {
    setPreviewZoom(1);
  }, []);

  const resetPreviewPan = useCallback(() => {
    setPreviewPanX(0);
    setPreviewPanY(0);
  }, []);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      if (!workspaceDraggingRef.current) {
        return;
      }

      const grid = workspaceGridRef.current;
      if (!grid) {
        return;
      }

      const rect = grid.getBoundingClientRect();
      const dividerWidth = 12;
      const availableWidth = rect.width - dividerWidth;
      if (availableWidth <= 0) {
        return;
      }

      const nextSplit = (event.clientX - rect.left - dividerWidth / 2) / availableWidth;
      const clampedSplit = Math.min(1, Math.max(0, nextSplit));
      setWorkspaceSplit(clampedSplit <= 0.03 ? 0 : clampedSplit >= 0.97 ? 1 : clampedSplit);
    };

    const stopDragging = () => {
      workspaceDraggingRef.current = false;
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopDragging);
    window.addEventListener("pointercancel", stopDragging);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopDragging);
      window.removeEventListener("pointercancel", stopDragging);
    };
  }, []);

  const isMainCollapsed = workspaceSplit <= 0;
  const isPreviewCollapsed = workspaceSplit >= 1;
  const workspaceGridColumns = isMainCollapsed
    ? "12px minmax(0, 1fr)"
    : isPreviewCollapsed
      ? "minmax(0, 1fr) 12px"
    : `minmax(0, ${workspaceSplit}fr) 12px minmax(0, ${1 - workspaceSplit}fr)`;

  const onPreviewWheel = useCallback((event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    setPreviewZoom((current) => {
      const nextZoom = adjustPreviewZoomForWheel(current, event.deltaY);
      if (nextZoom <= 1) {
        setPreviewPanX(0);
        setPreviewPanY(0);
      }
      return nextZoom;
    });
  }, []);

  const onPreviewPointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (previewZoom <= 1) {
      return;
    }
    event.preventDefault();
    previewDraggingRef.current = true;
    previewDragStartRef.current = { x: event.clientX, y: event.clientY };
    previewDragOriginRef.current = { x: previewPanX, y: previewPanY };
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [previewPanX, previewPanY, previewZoom]);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      if (!previewDraggingRef.current) {
        return;
      }

      const nextPan = applyPreviewDrag(
        previewDragOriginRef.current,
        event.clientX - previewDragStartRef.current.x,
        event.clientY - previewDragStartRef.current.y,
      );
      setPreviewPanX(nextPan.x);
      setPreviewPanY(nextPan.y);
    };

    const stopPreviewDragging = () => {
      previewDraggingRef.current = false;
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopPreviewDragging);
    window.addEventListener("pointercancel", stopPreviewDragging);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopPreviewDragging);
      window.removeEventListener("pointercancel", stopPreviewDragging);
    };
  }, []);

  const onDividerPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    workspaceDraggingRef.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  return {
    isMobileWorkspace,
    workspaceGridRef,
    workspaceGridColumns,
    workspaceSplit,
    previewZoom,
    previewPanX,
    previewPanY,
    isMainCollapsed,
    isPreviewCollapsed,
    setWorkspaceSplit,
    resetPreviewZoom,
    resetPreviewPan,
    onPreviewWheel,
    onPreviewPointerDown,
    onDividerPointerDown,
  };
}
