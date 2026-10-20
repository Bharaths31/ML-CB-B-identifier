import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import ImageCropper from "./ImageCropper.jsx";
import { QualityPill } from "./QualityBadge.jsx";
import { useI18n } from "../i18n/I18nProvider.jsx";
import {
  QUALITY_SAMPLE_SIZE,
  ZOOM_MAX,
  analyzeCanvas,
  captureVideoFrame,
  clamp,
  clampPan,
  computeSourceRect,
  drawCropToCanvas,
} from "../utils/cropAndQuality.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const stageSide = (w, h) => Math.min(w, h) * 0.78;

function qualityFromFrame(frame, rect) {
  try {
    const canvas = drawCropToCanvas(frame, rect, QUALITY_SAMPLE_SIZE);
    return analyzeCanvas(canvas);
  } catch {
    return null;
  }
}

/**
 * Live camera with an authoritative 1:1 target box. The square the user frames
 * is exactly what gets cropped, exported, and sent to /api/predict.
 */
export default function CameraCapture({ onCapture, onClose }) {
  const { t } = useI18n();
  const videoRef = useRef(null);
  const stageRef = useRef(null);
  const streamRef = useRef(null);
  const capturingRef = useRef(false);

  const [error, setError] = useState(null);
  const [facingMode, setFacingMode] = useState("environment");
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [isStarting, setIsStarting] = useState(true);
  const [stage, setStage] = useState({ w: 0, h: 0 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [quality, setQuality] = useState(null);
  const [readyStreak, setReadyStreak] = useState(0);
  const [autoCapture, setAutoCapture] = useState(false);
  const [burstMode, setBurstMode] = useState(false);
  const [torchSupported, setTorchSupported] = useState(false);
  const [torchOn, setTorchOn] = useState(false);
  const [flash, setFlash] = useState(false);
  const [captured, setCaptured] = useState(null);

  // Keep translations fresh without restarting the stream on language change.
  const tRef = useRef(t);
  useEffect(() => {
    tRef.current = t;
  }, [t]);

  // Refs keep the quality interval from closing over stale gesture state.
  const liveState = useRef({ zoom: 1, pan: { x: 0, y: 0 }, facingMode: "environment", autoCapture: false });
  useEffect(() => {
    liveState.current = { zoom, pan, facingMode, autoCapture };
  }, [zoom, pan, facingMode, autoCapture, burstMode]);

  // Enumerate cameras (best-effort).
  useEffect(() => {
    let cancelled = false;
    navigator.mediaDevices
      ?.enumerateDevices?.()
      .then((devices) => {
        if (cancelled) return;
        setHasMultipleCameras(devices.filter((d) => d.kind === "videoinput").length > 1);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // Start / restart the stream when the facing mode changes.
  useEffect(() => {
    let cancelled = false;
    setIsStarting(true);
    setError(null);
    setTorchSupported(false);
    setTorchOn(false);

    const startCamera = async () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      try {
        const constraints = {
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        };
        let stream;
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
        const track = stream.getVideoTracks?.()[0];
        const caps = track?.getCapabilities?.();
        setTorchSupported(Boolean(caps?.torch));
        setIsStarting(false);
      } catch (err) {
        if (cancelled) return;
        const name = err?.name || "";
        setError(
          name === "NotAllowedError" || name === "PermissionDeniedError"
            ? tRef.current("camera.error.permission")
            : name === "NotFoundError"
            ? tRef.current("camera.error.notFound")
            : `${tRef.current("camera.error.generic")} ${err?.message || ""}`.trim()
        );
        setIsStarting(false);
      }
    };

    startCamera();

    return () => {
      cancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
    // Intentionally not re-running on language change: a language switch must
    // not restart the camera stream. Error copy goes through tRef.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [facingMode]);

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
  }, [captured]);

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

  const rectForVideo = useCallback(
    (video, mirrored) => {
      if (!video || !video.videoWidth || !frame.rl) return null;
      return computeSourceRect({
        viewportW: frame.vw,
        viewportH: frame.vh,
        sourceW: video.videoWidth,
        sourceH: video.videoHeight,
        reticleX: frame.rx,
        reticleY: frame.ry,
        reticleSize: frame.rl,
        zoom,
        panX: pan.x,
        panY: pan.y,
        mirrored,
      });
    },
    [frame, zoom, pan.x, pan.y]
  );

  const applyZoom = useCallback(
    (nextZoom) => {
      const z = clamp(nextZoom, 1, ZOOM_MAX);
      setZoom(z);
      setPan((prev) =>
        clampPan({
          panX: prev.x,
          panY: prev.y,
          zoom: z,
          viewportW: frame.vw,
          viewportH: frame.vh,
          sourceW: videoRef.current?.videoWidth || 1,
          sourceH: videoRef.current?.videoHeight || 1,
        })
      );
    },
    [frame.vw, frame.vh]
  );

  // Native wheel zoom.
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

  const captureRawFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return null;
    const canvas = captureVideoFrame(video, facingMode === "user");
    return {
      canvas,
      w: canvas.width,
      h: canvas.height,
      url: canvas.toDataURL("image/jpeg", 0.95),
    };
  }, [facingMode]);

  const doCapture = useCallback(async () => {
    if (capturingRef.current) return;
    capturingRef.current = true;
    setFlash(true);
    setTimeout(() => setFlash(false), 220);

    try {
      const first = captureRawFrame();
      if (!first) return;

      const rect = computeSourceRect({
        viewportW: frame.vw,
        viewportH: frame.vh,
        sourceW: first.w,
        sourceH: first.h,
        reticleX: frame.rx,
        reticleY: frame.ry,
        reticleSize: frame.rl,
        zoom: liveState.current.zoom,
        panX: liveState.current.pan.x,
        panY: liveState.current.pan.y,
        mirrored: false, // frame already mirrored at capture time
      });

      let best = { frame: first, quality: qualityFromFrame(first.canvas, rect) };

      if (burstMode) {
        for (let i = 0; i < 2; i++) {
          await sleep(180);
          const next = captureRawFrame();
          if (!next) continue;
          const nextRect = computeSourceRect({
            viewportW: frame.vw,
            viewportH: frame.vh,
            sourceW: next.w,
            sourceH: next.h,
            reticleX: frame.rx,
            reticleY: frame.ry,
            reticleSize: frame.rl,
            zoom: liveState.current.zoom,
            panX: liveState.current.pan.x,
            panY: liveState.current.pan.y,
            mirrored: false,
          });
          const q = qualityFromFrame(next.canvas, nextRect);
          // Burst mode picks the sharpest frame, as the UI label promises.
          if ((q?.sharpnessScore || 0) > (best.quality?.sharpnessScore || 0)) {
            best = { frame: next, quality: q };
          }
        }
      }

      setCaptured({ url: best.frame.url, source: { w: best.frame.w, h: best.frame.h } });
      setQuality(best.quality);
      setReadyStreak(0);
    } finally {
      capturingRef.current = false;
    }
  }, [burstMode, captureRawFrame, frame]);

  // Live quality gate + auto-capture.
  useEffect(() => {
    if (captured || error) return undefined;
    const timer = setInterval(() => {
      const video = videoRef.current;
      const state = liveState.current;
      const rect = rectForVideo(video, state.facingMode === "user");
      if (!rect) return;
      try {
        const canvas = drawCropToCanvas(video, rect, QUALITY_SAMPLE_SIZE);
        const report = analyzeCanvas(canvas);
        setQuality(report);
        setReadyStreak((prev) => {
          const next = report.isReady ? prev + 1 : 0;
          if (state.autoCapture && next >= 3 && !capturingRef.current) {
            doCapture();
            return 0;
          }
          return next;
        });
      } catch {
        /* frame not ready */
      }
    }, 250); // 4 Hz; three READY checks = ~750 ms before auto-capture
    return () => clearInterval(timer);
  }, [captured, error, rectForVideo, doCapture]);

  const pointerRef = useRef(null);

  const handlePointerDown = (e) => {
    stageRef.current?.setPointerCapture?.(e.pointerId);
    pointerRef.current = {
      id: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      panX: pan.x,
      panY: pan.y,
      moved: false,
    };
  };

  const handlePointerMove = (e) => {
    const p = pointerRef.current;
    if (!p || p.id !== e.pointerId) return;
    const dx = e.clientX - p.startX;
    const dy = e.clientY - p.startY;
    if (Math.abs(dx) + Math.abs(dy) > 4) p.moved = true;
    setPan(
      clampPan({
        panX: p.panX + dx,
        panY: p.panY + dy,
        zoom,
        viewportW: frame.vw,
        viewportH: frame.vh,
        sourceW: videoRef.current?.videoWidth || 1,
        sourceH: videoRef.current?.videoHeight || 1,
      })
    );
  };

  const handlePointerUp = () => {
    pointerRef.current = null;
  };

  const toggleTorch = async () => {
    const track = streamRef.current?.getVideoTracks?.()[0];
    if (!track?.applyConstraints) return;
    try {
      await track.applyConstraints({ advanced: [{ torch: !torchOn }] });
      setTorchOn((v) => !v);
    } catch {
      setTorchSupported(false);
    }
  };

  const confirmCrop = (file, dataUrl, report) => {
    onCapture?.(file, dataUrl, report);
    onClose?.();
  };

  const offsetX = pan.x;
  const offsetY = pan.y;

  return (
    <div className="camera-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="camera-modal modern-camera-modal" onClick={(e) => e.stopPropagation()}>
        <div className="camera-modal-header">
          <div className="camera-title-wrap">
            {!captured && <span className="camera-pulse-dot" />}
            <h3>{captured ? t("camera.review") : t("camera.title")}</h3>
          </div>
          <button className="camera-close-icon" onClick={onClose} title={t("actions.close")}>
            ✕
          </button>
        </div>

        {captured ? (
          <div className="camera-crop-body">
            <ImageCropper
              imageUrl={captured.url}
              fileName="live_capture.jpg"
              onConfirm={confirmCrop}
              onCancel={() => setCaptured(null)}
              confirmLabel={t("actions.useThis")}
              cancelLabel={t("actions.retake")}
            />
          </div>
        ) : (
          <>
            <div className="camera-viewport-wrap">
              {flash && <div className="camera-flash" />}
              {error ? (
                <div className="camera-error-container">
                  <span className="camera-error-icon">📷</span>
                  <p className="camera-error-msg">{error}</p>
                  <button
                    className="btn ghost btn-sm"
                    onClick={() => setFacingMode((f) => (f === "environment" ? "user" : "environment"))}
                  >
                    🔄 {t("camera.error.retry")}
                  </button>
                </div>
              ) : (
                <div
                  className="camera-stage"
                  ref={stageRef}
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  onPointerCancel={handlePointerUp}
                >
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="camera-video"
                    style={{
                      transform: `translate(${offsetX}px, ${offsetY}px) scale(${zoom}) ${
                        facingMode === "user" ? "scaleX(-1)" : ""
                      }`,
                    }}
                  />

                  {isStarting && (
                    <div className="camera-loading-overlay">
                      <div className="camera-spinner" />
                      <span>{t("camera.starting")}</span>
                    </div>
                  )}

                  {frame.rl > 0 && (
                    <>
                      <div
                        className="camera-target"
                        style={{
                          left: frame.rx,
                          top: frame.ry,
                          width: frame.rl,
                          height: frame.rl,
                        }}
                      >
                        <span className="camera-target-corner tl" />
                        <span className="camera-target-corner tr" />
                        <span className="camera-target-corner bl" />
                        <span className="camera-target-corner br" />
                        <span className="camera-target-label">
                          {t("camera.modelSees")} · 1:1
                        </span>
                      </div>
                      <div
                        className="camera-grid"
                        style={{ left: frame.rx, top: frame.ry, width: frame.rl, height: frame.rl }}
                      >
                        <span /><span /><span /><span />
                      </div>
                    </>
                  )}

                  <div className="camera-guidance">
                    <QualityPill quality={quality} />
                    {autoCapture && readyStreak > 0 && (
                      <span className="camera-streak" aria-label={t("camera.holdSteady")}>
                        <svg className="camera-countdown-ring" viewBox="0 0 36 36" aria-hidden="true">
                          <circle className="ring-bg" cx="18" cy="18" r="15" />
                          <circle
                            className="ring-fg"
                            cx="18"
                            cy="18"
                            r="15"
                            style={{
                              strokeDasharray: 94.25,
                              strokeDashoffset: 94.25 * (1 - readyStreak / 3),
                            }}
                          />
                        </svg>
                        <span>{readyStreak}/3</span>
                      </span>
                    )}
                  </div>

                  <div className="camera-zoom-row">
                    <span>{t("camera.zoom")}</span>
                    <input
                      type="range"
                      min="1"
                      max={ZOOM_MAX}
                      step="0.05"
                      value={zoom}
                      onChange={(e) => applyZoom(Number(e.target.value))}
                      aria-label={t("camera.zoom")}
                    />
                    <span className="camera-zoom-value">{zoom.toFixed(1)}×</span>
                  </div>
                </div>
              )}
            </div>

            <div className="camera-controls-footer">
              <div className="camera-toggles-row">
                {hasMultipleCameras && (
                  <button
                    type="button"
                    className="camera-toggle-btn"
                    onClick={() => setFacingMode((f) => (f === "environment" ? "user" : "environment"))}
                    title={t("camera.flip")}
                  >
                    🔄 {t("camera.flip")}
                  </button>
                )}
                {torchSupported && (
                  <button
                    type="button"
                    className={`camera-toggle-btn ${torchOn ? "is-on" : ""}`}
                    onClick={toggleTorch}
                    title={t("camera.torch")}
                  >
                    💡 {torchOn ? t("camera.torchOn") : t("camera.torch")}
                  </button>
                )}
                <button
                  type="button"
                  className={`camera-toggle-btn ${autoCapture ? "is-on" : ""}`}
                  onClick={() => setAutoCapture((v) => !v)}
                  aria-pressed={autoCapture}
                >
                  ⏱ {t("camera.auto")}
                </button>
                <button
                  type="button"
                  className={`camera-toggle-btn ${burstMode ? "is-on" : ""}`}
                  onClick={() => setBurstMode((v) => !v)}
                  aria-pressed={burstMode}
                >
                  ⧉ {t("camera.burst")}
                </button>
              </div>

              <div className="camera-live-actions">
                <button
                  type="button"
                  className="camera-shutter-btn"
                  onClick={doCapture}
                  disabled={Boolean(error) || isStarting}
                  title={t("camera.shutter")}
                  aria-label={t("camera.shutter")}
                >
                  <span className="shutter-inner" />
                </button>
                <button type="button" className="btn ghost camera-cancel-btn" onClick={onClose}>
                  {t("actions.cancel")}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
