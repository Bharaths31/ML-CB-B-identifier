import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  EXPORT_SIZE,
  QUALITY_SAMPLE_SIZE,
  analyzeCanvas,
  clamp,
  clampPan,
  computeSourceRect,
  drawCropToCanvas,
  makeCroppedFile,
} from "../utils/cropAndQuality.js";
import { useI18n } from "../i18n/I18nProvider.jsx";
import QualityBadge from "./QualityBadge.jsx";

const stageSide = (w, h) => Math.min(w, h) * 0.78;

/**
 * Interactive 1:1 cropper. The square is authoritative: confirming exports
 * exactly the square region to a square JPEG, so the server-side center crop
 * is a no-op.
 */
export default function ImageCropper({
  imageUrl,
  fileName = "capture.jpg",
  onConfirm,
  onCancel,
  confirmLabel,
  cancelLabel,
  maxZoom = 3,
  title,
  hint,
}) {
  const { t } = useI18n();
  const stageRef = useRef(null);
  const imgRef = useRef(null);
  const pointers = useRef(new Map());
  const gesture = useRef(null);

  const [stage, setStage] = useState({ w: 0, h: 0 });
  const [source, setSource] = useState({ w: 0, h: 0 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [quality, setQuality] = useState(null);
  const [busy, setBusy] = useState(false);

  // Reset the viewport whenever the source changes.
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setQuality(null);
  }, [imageUrl]);

  // The cropper can sit below the fold in the upload card; bring it into view.
  useEffect(() => {
    stageRef.current?.scrollIntoView?.({ block: "center", inline: "nearest" });
  }, [imageUrl]);

  useLayoutEffect(() => {
    const node = stageRef.current;
    if (!node) return undefined;
    const measure = () => {
      const rect = node.getBoundingClientRect();
      setStage({ w: rect.width, h: rect.height });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    window.addEventListener("orientationchange", measure);
    return () => {
      observer.disconnect();
      window.removeEventListener("orientationchange", measure);
    };
  }, []);

  const side = stageSide(stage.w, stage.h);
  const frame = useMemo(
    () => ({
      vw: stage.w,
      vh: stage.h,
      rx: (stage.w - side) / 2,
      ry: (stage.h - side) / 2,
      rl: side,
    }),
    [stage.w, stage.h, side]
  );

  const rect = useMemo(() => {
    if (!source.w || !source.h || !frame.rl) return null;
    return computeSourceRect({
      viewportW: frame.vw,
      viewportH: frame.vh,
      sourceW: source.w,
      sourceH: source.h,
      reticleX: frame.rx,
      reticleY: frame.ry,
      reticleSize: frame.rl,
      zoom,
      panX: pan.x,
      panY: pan.y,
    });
  }, [frame, source.w, source.h, zoom, pan.x, pan.y]);

  // Live quality on the authoritative square, debounced during gestures.
  useEffect(() => {
    if (!rect || !imgRef.current) return undefined;
    const timer = setTimeout(() => {
      try {
        const canvas = drawCropToCanvas(imgRef.current, rect, QUALITY_SAMPLE_SIZE);
        setQuality(analyzeCanvas(canvas));
      } catch {
        /* image not ready yet */
      }
    }, 140);
    return () => clearTimeout(timer);
  }, [rect]);

  const applyZoom = useCallback(
    (nextZoom) => {
      const z = clamp(nextZoom, 1, maxZoom);
      setZoom(z);
      setPan((prev) =>
        clampPan({
          panX: prev.x,
          panY: prev.y,
          zoom: z,
          viewportW: frame.vw,
          viewportH: frame.vh,
          sourceW: source.w,
          sourceH: source.h,
          reticleSize: frame.rl,
        })
      );
    },
    [frame.vw, frame.vh, frame.rl, source.w, source.h, maxZoom]
  );

  // Native wheel listener so preventDefault works (React wheel is passive).
  useEffect(() => {
    const node = stageRef.current;
    if (!node) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      applyZoom(zoom * (e.deltaY < 0 ? 1.08 : 0.92));
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, [applyZoom, zoom]);

  const handlePointerDown = (e) => {
    stageRef.current?.setPointerCapture?.(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size === 1) {
      gesture.current = {
        mode: "pan",
        startX: e.clientX,
        startY: e.clientY,
        panX: pan.x,
        panY: pan.y,
      };
    } else if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      gesture.current = {
        mode: "pinch",
        startDist: Math.hypot(a.x - b.x, a.y - b.y),
        startZoom: zoom,
      };
    }
  };

  const handlePointerMove = (e) => {
    if (!pointers.current.has(e.pointerId)) return;
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const g = gesture.current;
    if (!g) return;

    if (g.mode === "pan" && pointers.current.size === 1) {
      const next = clampPan({
        panX: g.panX + (e.clientX - g.startX),
        panY: g.panY + (e.clientY - g.startY),
        zoom,
        viewportW: frame.vw,
        viewportH: frame.vh,
        sourceW: source.w,
        sourceH: source.h,
        reticleSize: frame.rl,
      });
      setPan(next);
    } else if (g.mode === "pinch" && pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      if (g.startDist > 0) {
        applyZoom(g.startZoom * (dist / g.startDist));
      }
    }
  };

  const endPointer = (e) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size === 0) {
      gesture.current = null;
    } else if (pointers.current.size === 1) {
      const [remaining] = [...pointers.current.values()];
      gesture.current = {
        mode: "pan",
        startX: remaining.x,
        startY: remaining.y,
        panX: pan.x,
        panY: pan.y,
      };
    }
  };

  const handleConfirm = async () => {
    if (!rect || !imgRef.current) return;
    setBusy(true);
    try {
      const base = (fileName || "capture").replace(/\.[^.]+$/, "");
      const result = await makeCroppedFile(imgRef.current, rect, {
        size: EXPORT_SIZE,
        filename: `crop_${base}_${Date.now()}.jpg`,
        type: "image/jpeg",
        quality: 0.92,
      });
      onConfirm?.(result.file, result.dataUrl, quality);
    } finally {
      setBusy(false);
    }
  };

  const offsetX = pan.x;
  const offsetY = pan.y;

  return (
    <div className="cropper-wrap">
      <div className="cropper-head">
        <h4 className="cropper-title">{title || t("camera.cropTitle")}</h4>
        <p className="cropper-hint">{hint || t("camera.cropHint")}</p>
      </div>

      <div
        className="crop-stage"
        ref={stageRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endPointer}
        onPointerCancel={endPointer}
        role="application"
        aria-label={t("camera.cropHint")}
      >
        <img
          ref={imgRef}
          src={imageUrl}
          alt=""
          draggable={false}
          className="crop-media"
          style={{ transform: `translate(${offsetX}px, ${offsetY}px) scale(${zoom})` }}
          onLoad={(e) =>
            setSource({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })
          }
        />

        {frame.rl > 0 && (
          <>
            <div
              className="crop-target"
              style={{
                left: frame.rx,
                top: frame.ry,
                width: frame.rl,
                height: frame.rl,
              }}
            >
              <span className="crop-corner tl" />
              <span className="crop-corner tr" />
              <span className="crop-corner bl" />
              <span className="crop-corner br" />
              <span className="crop-target-label">
                {t("camera.modelSees")} · 1:1 · 260×260
              </span>
            </div>
            <div className="crop-grid" style={{ left: frame.rx, top: frame.ry, width: frame.rl, height: frame.rl }}>
              <span /><span /><span /><span />
            </div>
          </>
        )}

        <div className="crop-zoom-control">
          <label htmlFor="crop-zoom">
            {t("camera.zoom")} {zoom.toFixed(1)}×
          </label>
          <input
            id="crop-zoom"
            type="range"
            min="1"
            max={maxZoom}
            step="0.05"
            value={zoom}
            onChange={(e) => applyZoom(Number(e.target.value))}
          />
          <button
            type="button"
            className="crop-reset-btn"
            onClick={() => {
              applyZoom(1);
              setPan({ x: 0, y: 0 });
            }}
          >
            {t("actions.reset")}
          </button>
        </div>
      </div>

      <QualityBadge quality={quality} />

      <div className="cropper-actions">
        <button type="button" className="btn btn-outline" onClick={onCancel} disabled={busy}>
          {cancelLabel || t("actions.cancel")}
        </button>
        <button type="button" className="btn btn-primary" onClick={handleConfirm} disabled={busy || !rect}>
          {busy ? t("upload.processing") : confirmLabel || t("actions.useThis")}
        </button>
      </div>
    </div>
  );
}
