import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import ImageCropper from "./ImageCropper.jsx";
import QualityBadge, { QualityPill } from "./QualityBadge.jsx";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { analyzeCanvas, captureVideoFrame } from "../utils/cropAndQuality.js";
import {
  formatBytes,
  optimizeImageTo260,
  optimizeSourceTo260,
} from "../utils/imageOptimizer.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function analyzeFullFrame(source) {
  const canvas = document.createElement("canvas");
  canvas.width = 130;
  canvas.height = 130;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, 130, 130);
  return analyzeCanvas(canvas);
}

/**
 * Full-frame camera. The live view shows the entire sensor frame
 * (`object-fit: contain`); capture sends the full frame through the
 * zero-crop 260x260 optimizer. The optional "Adjust framing" review still
 * uses the shared cropper when a user wants manual control.
 */
export default function CameraCapture({ onCapture, onClose }) {
  const { t } = useI18n();
  const videoRef = useRef(null);
  const stageRef = useRef(null);
  const streamRef = useRef(null);
  const capturingRef = useRef(false);
  const rawCanvasRef = useRef(null);
  const rawFileRef = useRef(null);

  const [error, setError] = useState(null);
  const [facingMode, setFacingMode] = useState("environment");
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [isStarting, setIsStarting] = useState(true);
  const [stage, setStage] = useState({ w: 0, h: 0 });
  const [videoDims, setVideoDims] = useState({ w: 0, h: 0 });
  const [quality, setQuality] = useState(null);
  const [readyStreak, setReadyStreak] = useState(0);
  const [autoCapture, setAutoCapture] = useState(false);
  const [burstMode, setBurstMode] = useState(false);
  const [torchSupported, setTorchSupported] = useState(false);
  const [torchOn, setTorchOn] = useState(false);
  const [flash, setFlash] = useState(false);
  const [captured, setCaptured] = useState(null);
  const [fitMode, setFitMode] = useState("pad");
  const [busy, setBusy] = useState(false);
  const [adjusting, setAdjusting] = useState(false);

  const tRef = useRef(t);
  useEffect(() => {
    tRef.current = t;
  }, [t]);

  const liveState = useRef({ autoCapture: false, burstMode: false });
  useEffect(() => {
    liveState.current = { autoCapture, burstMode };
  }, [autoCapture, burstMode]);

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
          // Fallback for browsers that do not fire loadedmetadata reliably.
          setTimeout(() => {
            const video = videoRef.current;
            if (video?.videoWidth) {
              setVideoDims({ w: video.videoWidth, h: video.videoHeight });
            }
          }, 300);
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
  }, [captured, adjusting]);

  const videoRect = useMemo(() => {
    if (!stage.w || !stage.h || !videoDims.w || !videoDims.h) return null;
    const scale = Math.min(stage.w / videoDims.w, stage.h / videoDims.h);
    const width = videoDims.w * scale;
    const height = videoDims.h * scale;
    return {
      left: (stage.w - width) / 2,
      top: (stage.h - height) / 2,
      width,
      height,
    };
  }, [stage, videoDims]);

  const optimizeCanvas = useCallback(
    (canvas, mode, originalDataUrl) =>
      optimizeSourceTo260(canvas, canvas.width, canvas.height, {
        fitMode: mode,
        padStyle: "imagenet_mean",
        allowUpscale: false,
        originalName: `camera_${Date.now()}.jpg`,
        originalBytes: 0,
        originalDataUrl: originalDataUrl || canvas.toDataURL("image/jpeg", 0.92),
      }),
    []
  );

  const doCapture = useCallback(async () => {
    if (capturingRef.current) return;
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;

    capturingRef.current = true;
    setBusy(true);
    setFlash(true);
    setTimeout(() => setFlash(false), 220);

    try {
      const first = captureVideoFrame(video, facingMode === "user");
      let best = { canvas: first, quality: analyzeFullFrame(first) };

      if (liveState.current.burstMode) {
        for (let i = 0; i < 2; i++) {
          await sleep(180);
          const next = captureVideoFrame(video, facingMode === "user");
          const q = analyzeFullFrame(next);
          // The UI promises "sharpest frame", not "best exposed frame".
          if ((q?.sharpnessScore || 0) > (best.quality?.sharpnessScore || 0)) {
            best = { canvas: next, quality: q };
          }
        }
      }

      const rawUrl = best.canvas.toDataURL("image/jpeg", 0.92);
      rawCanvasRef.current = best.canvas;
      rawFileRef.current = null;
      const optimized = await optimizeCanvas(best.canvas, fitMode, rawUrl);
      setCaptured({
        rawUrl,
        optimized,
        quality: best.quality,
        fitMode,
        manuallyCropped: false,
      });
      setReadyStreak(0);
    } catch (err) {
      setError(err?.message || "Capture failed");
    } finally {
      capturingRef.current = false;
      setBusy(false);
    }
  }, [facingMode, fitMode, optimizeCanvas]);

  // Live quality gate on the full frame at 4 Hz.
  useEffect(() => {
    if (captured || adjusting || error) return undefined;
    const timer = setInterval(() => {
      const video = videoRef.current;
      if (!video || !video.videoWidth) return;
      try {
        const report = analyzeFullFrame(video);
        setQuality(report);
        setReadyStreak((prev) => {
          const next = report.isReady ? prev + 1 : 0;
          if (liveState.current.autoCapture && next >= 3 && !capturingRef.current) {
            doCapture();
            return 0;
          }
          return next;
        });
      } catch {
        /* frame not ready */
      }
    }, 250);
    return () => clearInterval(timer);
  }, [captured, adjusting, error, doCapture]);

  const changeFitMode = async (mode) => {
    if (mode === fitMode || busy) return;
    setFitMode(mode);
    if (!captured) return;
    setBusy(true);
    try {
      let optimized;
      if (rawCanvasRef.current) {
        optimized = await optimizeCanvas(rawCanvasRef.current, mode, captured.rawUrl);
      } else if (rawFileRef.current) {
        optimized = await optimizeImageTo260(rawFileRef.current, {
          fitMode: mode,
          padStyle: "imagenet_mean",
          allowUpscale: false,
          originalDataUrl: captured.rawUrl,
        });
      }
      if (optimized) {
        setCaptured((current) => (current ? { ...current, optimized, fitMode: mode } : current));
      }
    } catch (err) {
      setError(err?.message || "Optimization failed");
    } finally {
      setBusy(false);
    }
  };

  const handleAdjustConfirm = async (file, dataUrl, report) => {
    setAdjusting(false);
    setBusy(true);
    try {
      const optimized = await optimizeImageTo260(file, {
        fitMode,
        padStyle: "imagenet_mean",
        allowUpscale: false,
        originalDataUrl: dataUrl,
      });
      optimized.meta.manuallyCropped = true;
      rawCanvasRef.current = null;
      rawFileRef.current = file;
      setCaptured({
        rawUrl: dataUrl,
        optimized,
        quality: report || captured?.quality || null,
        fitMode,
        manuallyCropped: true,
      });
    } catch (err) {
      setError(err?.message || "Optimization failed");
    } finally {
      setBusy(false);
    }
  };

  const retake = () => {
    setCaptured(null);
    setAdjusting(false);
    rawCanvasRef.current = null;
    rawFileRef.current = null;
    setReadyStreak(0);
  };

  const confirmCapture = () => {
    const optimized = captured?.optimized;
    if (!optimized || busy) return;
    onCapture?.(optimized.file, optimized.dataUrl, optimized.meta, captured.quality);
    onClose?.();
  };

  const toggleTorch = async () => {
    const track = streamRef.current?.getVideoTracks?.()[0];
    if (!track?.applyConstraints) return;
    try {
      await track.applyConstraints({ advanced: [{ torch: !torchOn }] });
      setTorchOn((value) => !value);
    } catch {
      setTorchSupported(false);
    }
  };

  const capturedMeta = captured?.optimized?.meta;
  const telemetry = capturedMeta
    ? t("camera.telemetry", {
        w: capturedMeta.originalWidth,
        h: capturedMeta.originalHeight,
        crop: captured.manuallyCropped
          ? t("camera.manualCrop")
          : t("optimizer.zeroCropPct"),
        size: formatBytes(capturedMeta.optimizedBytes),
      })
    : "";

  return (
    <div className="camera-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="camera-modal modern-camera-modal" onClick={(event) => event.stopPropagation()}>
        <div className="camera-modal-header">
          <div className="camera-title-wrap">
            {!captured && <span className="camera-pulse-dot" />}
            <h3>{captured ? t("camera.reviewTitle") : t("camera.title")}</h3>
          </div>
          <button className="camera-close-icon" onClick={onClose} title={t("actions.close")}>
            ✕
          </button>
        </div>

        {captured ? (
          adjusting ? (
            <div className="camera-crop-body">
              <ImageCropper
                imageUrl={captured.rawUrl}
                fileName="camera_frame.jpg"
                onConfirm={handleAdjustConfirm}
                onCancel={() => setAdjusting(false)}
                confirmLabel={t("actions.useThis")}
                cancelLabel={t("actions.back")}
              />
            </div>
          ) : (
            <div className="camera-review-panel">
              <div className="preview-260-frame">
                {captured.optimized?.dataUrl ? (
                  <img src={captured.optimized.dataUrl} alt="Optimized 260 by 260 model input" />
                ) : (
                  <div className="optimizer-loading">{t("optimizer.optimizing")}</div>
                )}
                <span className="preview-260-badge">260×260</span>
              </div>

              <div className="optimizer-bar">
                <span className="optimizer-badge">{telemetry}</span>
              </div>

              <div className="optimizer-controls" role="group" aria-label={t("optimizer.fitMode")}>
                <button
                  type="button"
                  className={`optimizer-mode-btn ${fitMode === "pad" ? "is-on" : ""}`}
                  onClick={() => changeFitMode("pad")}
                  disabled={busy}
                >
                  {t("optimizer.padMean")}
                </button>
                <button
                  type="button"
                  className={`optimizer-mode-btn ${fitMode === "stretch" ? "is-on" : ""}`}
                  onClick={() => changeFitMode("stretch")}
                  disabled={busy}
                >
                  {t("optimizer.stretch")}
                </button>
              </div>

              <QualityBadge quality={captured.quality} />

              {capturedMeta && (
                <div className="optimizer-meta-grid">
                  <div>
                    <span>{t("optimizer.original")}</span>
                    <strong>
                      {capturedMeta.originalWidth}×{capturedMeta.originalHeight}
                    </strong>
                  </div>
                  <div>
                    <span>{t("optimizer.modelInput")}</span>
                    <strong>260×260</strong>
                  </div>
                  <div>
                    <span>{t("optimizer.scaled")}</span>
                    <strong>
                      {capturedMeta.scaledInnerWidth}×{capturedMeta.scaledInnerHeight}
                    </strong>
                  </div>
                  <div>
                    <span>{t("optimizer.padding")}</span>
                    <strong>
                      {capturedMeta.padX}, {capturedMeta.padY}
                    </strong>
                  </div>
                  <div>
                    <span>{t("optimizer.bytes")}</span>
                    <strong>{formatBytes(capturedMeta.optimizedBytes)}</strong>
                  </div>
                  <div>
                    <span>{t("optimizer.time")}</span>
                    <strong>{capturedMeta.durationMs} ms</strong>
                  </div>
                </div>
              )}

              {error && <p className="camera-error-inline">{error}</p>}

              <div className="camera-review-actions">
                <button type="button" className="btn btn-outline" onClick={retake} disabled={busy}>
                  {t("actions.retake")}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setAdjusting(true)}
                  disabled={busy}
                >
                  {t("optimizer.adjust")}
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={confirmCapture}
                  disabled={busy || !captured.optimized}
                >
                  {t("actions.useThis")}
                </button>
              </div>

              {busy && (
                <div className="optimizer-busy-overlay">
                  <span className="camera-spinner" />
                  <span>{t("optimizer.optimizing")}</span>
                </div>
              )}
            </div>
          )
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
                    onClick={() => setFacingMode((mode) => (mode === "environment" ? "user" : "environment"))}
                  >
                    🔄 {t("camera.error.retry")}
                  </button>
                </div>
              ) : (
                <div className="camera-stage camera-stage-fullframe" ref={stageRef}>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`camera-video camera-video-contain ${
                      facingMode === "user" ? "mirrored" : ""
                    }`}
                    onLoadedMetadata={(event) =>
                      setVideoDims({
                        w: event.currentTarget.videoWidth,
                        h: event.currentTarget.videoHeight,
                      })
                    }
                  />

                  {isStarting && (
                    <div className="camera-loading-overlay">
                      <div className="camera-spinner" />
                      <span>{t("camera.starting")}</span>
                    </div>
                  )}

                  {videoRect && (
                    <div
                      className="camera-fullframe-guide"
                      style={{
                        left: videoRect.left,
                        top: videoRect.top,
                        width: videoRect.width,
                        height: videoRect.height,
                      }}
                    >
                      <span className="camera-target-corner tl" />
                      <span className="camera-target-corner tr" />
                      <span className="camera-target-corner bl" />
                      <span className="camera-target-corner br" />
                      <span className="camera-fullframe-badge">
                        {t("camera.fullFrameBadge")}
                      </span>
                    </div>
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
                </div>
              )}
            </div>

            <div className="camera-controls-footer">
              <div className="camera-toggles-row">
                {hasMultipleCameras && (
                  <button
                    type="button"
                    className="camera-toggle-btn"
                    onClick={() => setFacingMode((mode) => (mode === "environment" ? "user" : "environment"))}
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
                  onClick={() => setAutoCapture((value) => !value)}
                  aria-pressed={autoCapture}
                >
                  ⏱ {t("camera.auto")}
                </button>
                <button
                  type="button"
                  className={`camera-toggle-btn ${burstMode ? "is-on" : ""}`}
                  onClick={() => setBurstMode((value) => !value)}
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
                  disabled={Boolean(error) || isStarting || busy}
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
