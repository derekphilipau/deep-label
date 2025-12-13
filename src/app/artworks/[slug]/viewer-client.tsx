'use client';

import * as React from 'react';
import { ZoomIn, ZoomOut, Home, Maximize, Minimize } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { DetectedObject } from '@/lib/types';
import type OpenSeadragonType from 'openseadragon';

const TYPE_COLORS: Record<string, string> = {
  person: '#e63946',
  animal: '#2ec4b6',
  building: '#457b9d',
  landscape: '#f4a261',
  object: '#9b5de5',
  other: '#ffc300',
};

function normalizeType(value: string) {
  return (value || 'other').trim().toLowerCase();
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizeBox(box: number[]) {
  const x1 = clamp(Math.round(box[0] ?? 0), 0, 1000);
  const y1 = clamp(Math.round(box[1] ?? 0), 0, 1000);
  const x2 = clamp(Math.round(box[2] ?? 0), 0, 1000);
  const y2 = clamp(Math.round(box[3] ?? 0), 0, 1000);
  return {
    xmin: Math.min(x1, x2),
    ymin: Math.min(y1, y2),
    xmax: Math.max(x1, x2),
    ymax: Math.max(y1, y2),
  };
}

export default function ViewerClient({
  imageSrc,
  objects,
}: {
  imageSrc: string;
  objects: DetectedObject[];
}) {
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const hostRef = React.useRef<HTMLDivElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const viewerRef = React.useRef<OpenSeadragonType.Viewer | null>(null);
  const osdRef = React.useRef<typeof OpenSeadragonType | null>(null);

  const [showLabels, setShowLabels] = React.useState(true);
  const [showBoxes, setShowBoxes] = React.useState(true);
  const [showNavigator, setShowNavigator] = React.useState(false);
  const [hoveredIndex, setHoveredIndex] = React.useState<number | null>(null);
  const [osdLoaded, setOsdLoaded] = React.useState(false);
  const [isFullscreen, setIsFullscreen] = React.useState(false);

  const requestDrawRef = React.useRef<number | null>(null);
  const drawOverlayRef = React.useRef<() => void>(() => {});
  const labelHitRegionsRef = React.useRef<
    { index: number; x: number; y: number; w: number; h: number }[]
  >([]);

  const scheduleDraw = React.useCallback(() => {
    if (requestDrawRef.current != null) return;
    requestDrawRef.current = window.requestAnimationFrame(() => {
      requestDrawRef.current = null;
      drawOverlayRef.current();
    });
  }, []);

  const drawOverlay = React.useCallback(() => {
    const viewer = viewerRef.current;
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const OSD = osdRef.current;
    if (!viewer || !canvas || !container || !OSD) return;
    if (!viewer.world || viewer.world.getItemCount() === 0) return;

    const item = viewer.world.getItemAt(0);
    const contentSize = item.getContentSize();
    const imageWidth = contentSize.x;
    const imageHeight = contentSize.y;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();
    const cssWidth = Math.max(1, Math.floor(rect.width));
    const cssHeight = Math.max(1, Math.floor(rect.height));

    if (canvas.width !== Math.floor(cssWidth * dpr) || canvas.height !== Math.floor(cssHeight * dpr)) {
      canvas.width = Math.floor(cssWidth * dpr);
      canvas.height = Math.floor(cssHeight * dpr);
      canvas.style.width = `${cssWidth}px`;
      canvas.style.height = `${cssHeight}px`;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    if (!showBoxes) return;

    ctx.lineWidth = 2;
    ctx.font = '12px system-ui, -apple-system, Segoe UI, sans-serif';
    ctx.textBaseline = 'top';

    const labelHitRegions: { index: number; x: number; y: number; w: number; h: number }[] = [];

    for (let idx = 0; idx < objects.length; idx++) {
      const obj = objects[idx];
      const { xmin, ymin, xmax, ymax } = normalizeBox(obj.box_2d);
      const x1 = (xmin / 1000) * imageWidth;
      const y1 = (ymin / 1000) * imageHeight;
      const x2 = (xmax / 1000) * imageWidth;
      const y2 = (ymax / 1000) * imageHeight;

      const p1 = viewer.viewport.imageToViewerElementCoordinates(
        new OSD.Point(x1, y1)
      );
      const p2 = viewer.viewport.imageToViewerElementCoordinates(
        new OSD.Point(x2, y2)
      );

      const left = p1.x;
      const top = p1.y;
      const width = p2.x - p1.x;
      const height = p2.y - p1.y;

      if (width <= 1 || height <= 1) continue;

      const t = normalizeType(obj.type);
      const stroke = TYPE_COLORS[t] || TYPE_COLORS.other;
      const isHovered = hoveredIndex === idx;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = isHovered ? 4 : 2;
      if (isHovered) {
        ctx.fillStyle = 'rgba(255,255,255,0.06)';
        ctx.fillRect(left, top, width, height);
      }
      ctx.strokeRect(left, top, width, height);

      if (!showLabels) continue;
      if (width < 40 || height < 24) continue;

      const label = obj.label || '';
      if (!label) continue;

      const paddingX = 4;
      const paddingY = 2;
      const text = `${label}`;
      const metrics = ctx.measureText(text);
      const textW = Math.min(metrics.width, 260);
      const boxW = textW + paddingX * 2;
      const boxH = 12 + paddingY * 2;
      const x = clamp(left, 0, cssWidth - boxW);
      const y = clamp(top, 0, cssHeight - boxH);

      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.fillRect(x, y, boxW, boxH);
      ctx.fillStyle = '#fff';
      ctx.fillText(text, x + paddingX, y + paddingY);

      labelHitRegions.push({ index: idx, x, y, w: boxW, h: boxH });
    }

    labelHitRegionsRef.current = labelHitRegions;
  }, [objects, showBoxes, showLabels, hoveredIndex]);

  React.useEffect(() => {
    drawOverlayRef.current = drawOverlay;
  }, [drawOverlay]);

  // Dynamically import OpenSeadragon on client only
  React.useEffect(() => {
    let cancelled = false;
    import('openseadragon').then((mod) => {
      if (cancelled) return;
      osdRef.current = mod.default;
      setOsdLoaded(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  React.useEffect(() => {
    const OSD = osdRef.current;
    if (!OSD || !hostRef.current) return;

    // Tear down any existing viewer (e.g., hot reload)
    if (viewerRef.current) {
      viewerRef.current.destroy();
      viewerRef.current = null;
    }

    const viewer = OSD({
      element: hostRef.current,
      tileSources: {
        type: 'image',
        url: imageSrc,
      },
      crossOriginPolicy: 'Anonymous',
      showNavigationControl: false,
      showNavigator,
      zoomPerScroll: 1.2,
      minZoomLevel: 0.8,
      maxZoomLevel: 50,
      gestureSettingsMouse: {
        clickToZoom: true,
        dblClickToZoom: true,
        flickEnabled: true,
        pinchToZoom: true as any,
        scrollToZoom: true,
      } as any,
      gestureSettingsTouch: {
        clickToZoom: false,
        dblClickToZoom: true,
        flickEnabled: true,
        pinchToZoom: true,
      },
    });

    viewerRef.current = viewer;

    const onAny = () => scheduleDraw();
    viewer.addHandler('open', onAny);
    viewer.addHandler('resize', onAny);
    viewer.addHandler('animation', onAny);
    viewer.addHandler('animation-finish', onAny);
    viewer.addHandler('pan', onAny);
    viewer.addHandler('zoom', onAny);

    scheduleDraw();

    return () => {
      viewer.removeHandler('open', onAny);
      viewer.removeHandler('resize', onAny);
      viewer.removeHandler('animation', onAny);
      viewer.removeHandler('animation-finish', onAny);
      viewer.removeHandler('pan', onAny);
      viewer.removeHandler('zoom', onAny);
      viewer.destroy();
      viewerRef.current = null;
    };
  }, [imageSrc, scheduleDraw, showNavigator, osdLoaded]);

  React.useEffect(() => {
    scheduleDraw();
  }, [objects, showBoxes, showLabels, hoveredIndex, scheduleDraw]);

  React.useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const onMouseMove = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // Prefer top-most label: iterate in reverse draw order.
      const regions = labelHitRegionsRef.current;
      let nextHovered: number | null = null;
      for (let i = regions.length - 1; i >= 0; i--) {
        const r = regions[i];
        if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) {
          nextHovered = r.index;
          break;
        }
      }

      setHoveredIndex((prev) => (prev === nextHovered ? prev : nextHovered));
    };

    const onMouseLeave = () => setHoveredIndex(null);

    container.addEventListener('mousemove', onMouseMove, { passive: true });
    container.addEventListener('mouseleave', onMouseLeave, { passive: true });
    return () => {
      container.removeEventListener('mousemove', onMouseMove);
      container.removeEventListener('mouseleave', onMouseLeave);
    };
  }, []);

  const zoomIn = () => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.viewport.zoomBy(1.5);
    viewer.viewport.applyConstraints();
  };

  const zoomOut = () => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.viewport.zoomBy(0.67);
    viewer.viewport.applyConstraints();
  };

  const goHome = () => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.viewport.goHome(true);
  };

  const toggleFullscreen = () => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.setFullScreen(!isFullscreen);
    setIsFullscreen(!isFullscreen);
  };

  // Sync fullscreen state with actual fullscreen changes
  React.useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const handleFullScreen = (event: { fullScreen: boolean }) => {
      setIsFullscreen(event.fullScreen);
    };

    viewer.addHandler('full-screen', handleFullScreen);
    return () => {
      viewer.removeHandler('full-screen', handleFullScreen);
    };
  }, [osdLoaded]);

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between gap-2 border-b px-3 py-2">
        <div className="text-xs text-muted-foreground">
          Pinch/scroll to zoom • Drag to pan • Double-click to zoom
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={showBoxes}
              onChange={(e) => setShowBoxes(e.target.checked)}
            />
            Boxes
          </label>
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
            />
            Labels
          </label>
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={showNavigator}
              onChange={(e) => setShowNavigator(e.target.checked)}
            />
            Navigator
          </label>
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative h-[70vh] w-full overflow-hidden bg-black/5"
      >
        <div ref={hostRef} className="absolute inset-0" />
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0"
        />

        {/* Zoom Controls */}
        <div className="absolute bottom-4 right-4 flex flex-col gap-1 z-10">
          <Button
            variant="secondary"
            size="icon"
            onClick={zoomIn}
            title="Zoom in"
            className="h-9 w-9 shadow-md"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button
            variant="secondary"
            size="icon"
            onClick={zoomOut}
            title="Zoom out"
            className="h-9 w-9 shadow-md"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button
            variant="secondary"
            size="icon"
            onClick={goHome}
            title="Reset view"
            className="h-9 w-9 shadow-md"
          >
            <Home className="h-4 w-4" />
          </Button>
          <Button
            variant="secondary"
            size="icon"
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            className="h-9 w-9 shadow-md"
          >
            {isFullscreen ? (
              <Minimize className="h-4 w-4" />
            ) : (
              <Maximize className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
