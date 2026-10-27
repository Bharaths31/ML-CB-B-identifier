import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import ImageCropper from "./ImageCropper.jsx";
import QualityBadge, { QualityPill } from "./QualityBadge.jsx";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { analyzeCanvas, captureVideoFrame, QUALITY_THRESHOLDS } from "../utils/cropAndQuality.js";
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
  const [fitMode, setFitMode] = useState("stretch");
  const [claheOn, setClaheOn] = useState(false);
  const [silhouette, setSilhouette] = useState("off");
  const [silhouetteFlipped, setSilhouetteFlipped] = useState(false);
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
    (canvas, mode, originalDataUrl, clahe = false) =>
      optimizeSourceTo260(canvas, canvas.width, canvas.height, {
        fitMode: mode,
        padStyle: "imagenet_mean",
        allowUpscale: false,
        clahe,
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
      const optimized = await optimizeCanvas(best.canvas, fitMode, rawUrl, claheOn);
      setCaptured({
        rawUrl,
        originalRawUrl: rawUrl,
        optimized,
        quality: best.quality,
        fitMode,
        clahe: claheOn,
        manuallyCropped: false,
      });
      setReadyStreak(0);
    } catch (err) {
      setError(err?.message || "Capture failed");
    } finally {
      capturingRef.current = false;
      setBusy(false);
    }
  }, [facingMode, fitMode, claheOn, optimizeCanvas]);

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
      const originalRawUrl = captured.originalRawUrl || captured.rawUrl;
      // A manual crop wins for the current model input, but the cropper itself
      // always re-opens the original frame.
      if (rawFileRef.current) {
        optimized = await optimizeImageTo260(rawFileRef.current, {
          fitMode: mode,
          padStyle: "imagenet_mean",
          allowUpscale: false,
          clahe: claheOn,
          originalDataUrl: originalRawUrl,
        });
      } else if (rawCanvasRef.current) {
        optimized = await optimizeCanvas(rawCanvasRef.current, mode, originalRawUrl, claheOn);
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

  const changeClahe = async (next) => {
    setClaheOn(next);
    if (!captured || busy) return;
    setBusy(true);
    try {
      const originalRawUrl = captured.originalRawUrl || captured.rawUrl;
      let optimized;
      if (rawFileRef.current) {
        optimized = await optimizeImageTo260(rawFileRef.current, {
          fitMode,
          padStyle: "imagenet_mean",
          allowUpscale: false,
          clahe: next,
          originalDataUrl: originalRawUrl,
        });
      } else if (rawCanvasRef.current) {
        optimized = await optimizeCanvas(rawCanvasRef.current, fitMode, originalRawUrl, next);
      }
      if (optimized) {
        setCaptured((current) => (current ? { ...current, optimized, clahe: next } : current));
      }
    } catch (err) {
      setError(err?.message || "Shadow normalization failed");
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
        clahe: claheOn,
        originalDataUrl: dataUrl,
      });
      optimized.meta.manuallyCropped = true;
      rawFileRef.current = file;
      setCaptured((current) => ({
        ...(current || {}),
        rawUrl: current?.rawUrl || dataUrl,
        originalRawUrl: current?.originalRawUrl || dataUrl,
        optimized,
        quality: report || current?.quality || null,
        fitMode,
        manuallyCropped: true,
      }));
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

  const lightingKey = !quality
    ? null
    : quality.lightingState ||
      (quality.meanLuma < QUALITY_THRESHOLDS.darkMean
        ? "too_dark"
        : quality.meanLuma > QUALITY_THRESHOLDS.brightMean
        ? "overexposed"
        : "good");

  const lightingStatus = !quality
    ? { key: "unknown", label: "—" }
    : lightingKey === "too_dark"
    ? { key: "dark", label: t("camera.lightDark") }
    : lightingKey === "overexposed"
    ? { key: "overexposed", label: t("camera.lightBright") }
    : { key: "good", label: t("camera.lightGood") };

  const framingStatus = !quality
    ? { key: "unknown", label: "—" }
    : quality.framingState === "ready"
    ? { key: "ready", label: t("camera.frameAligned") }
    : quality.framingState === "edge_clipped"
    ? { key: "edge_clipped", label: t("camera.frameClipped") }
    : { key: "no_subject", label: t("camera.frameNoSubject") };

  const sharpnessStatus = !quality
    ? { key: "unknown", label: "—" }
    : quality.sharpnessState === "optimal"
    ? { key: "optimal", label: t("camera.sharpOptimal") }
    : { key: "blurry", label: t("camera.sharpBlurry") };

  const silhouetteLabel = silhouette === "buffalo"
    ? t("camera.silhouetteBuffalo")
    : silhouette === "cattle"
    ? t("camera.silhouetteCattle")
    : t("camera.silhouetteOff");

  const facingRight = !silhouetteFlipped;
  const headX = facingRight ? 615 : 185;
  const headY = 165;
  const humpX = facingRight ? 480 : 320;
  const humpY = 145;
  const dewlapX = facingRight ? 565 : 235;
  const dewlapY = 285;
  const tailX = facingRight ? 165 : 635;
  const tailY = 245;
  const isBuffaloGuide = silhouette === "buffalo";
  const guideLabel = isBuffaloGuide ? "━ Dorsal Ridge" : `▲ ${t("camera.landmarkHump")}`;
  const guideWidth = isBuffaloGuide ? 120 : 86;

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
                imageUrl={captured.originalRawUrl || captured.rawUrl}
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

              <div className="control-setting-row clahe-row">
                <div className="setting-info">
                  <span className="setting-label">{t("camera.clahe")}</span>
                  <span className="setting-subtext">{t("camera.claheHint")}</span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={claheOn}
                    onChange={(event) => changeClahe(event.target.checked)}
                    disabled={busy}
                  />
                  <span className="toggle-slider" />
                </label>
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
                      <svg
                        className={`camera-silhouette camera-hud ${silhouette === "off" ? "is-minimal" : ""}`}
                        viewBox="0 0 800 500"
                        preserveAspectRatio="xMidYMid meet"
                        aria-label={silhouette === "off" ? t("camera.minimalGrid") : `${silhouetteLabel} AR guide`}
                      >
                        <g className="hud-frame">
                          <rect className="hud-safe-frame" x="70" y="55" width="660" height="365" rx="16" />
                          <path className="hud-corner" d="M70 95 V71 Q70 55 86 55 H110" />
                          <path className="hud-corner" d="M690 55 H714 Q730 55 730 71 V95" />
                          <path className="hud-corner" d="M70 380 V404 Q70 420 86 420 H110" />
                          <path className="hud-corner" d="M690 420 H714 Q730 420 730 404 V380" />
                          <line className="hud-axis" x1="70" y1="175" x2="730" y2="175" />
                          <line className="hud-axis" x1="70" y1="335" x2="730" y2="335" />
                          <text className="hud-axis-label" x="742" y="179">DORSAL</text>
                          <text className="hud-axis-label" x="742" y="339">CHEST</text>
                        </g>

                        {silhouette !== "off" && (
                          <g className="hud-zones">
                            <path
                              className="hud-spine"
                              d={`M ${tailX} ${tailY} Q ${(tailX + humpX) / 2} ${tailY - 80} ${humpX} ${humpY} Q ${(humpX + headX) / 2} ${humpY - 24} ${headX} ${headY}`}
                            />

                            {/* Head & Horns */}
                            <circle className="hud-reticle" cx={headX} cy={headY} r="54" />
                            <path
                              className="hud-horn-guide"
                              d={
                                isBuffaloGuide
                                  ? `M ${headX - 30} ${headY - 20} Q ${headX - 66} ${headY - 32} ${headX - 44} ${headY + 12} M ${headX + 30} ${headY - 20} Q ${headX + 66} ${headY - 32} ${headX + 44} ${headY + 12}`
                                  : `M ${headX - 30} ${headY - 28} Q ${headX - 58} ${headY - 62} ${headX - 20} ${headY - 64} M ${headX + 30} ${headY - 28} Q ${headX + 58} ${headY - 62} ${headX + 20} ${headY - 64}`
                              }
                            />
                            <path
                              className="hud-crosshair"
                              d={`M ${headX - 14} ${headY + 6} H ${headX + 14} M ${headX} ${headY - 8} V ${headY + 20}`}
                            />
                            <g className="hud-label" transform={`translate(${headX}, ${headY - 66})`}>
                              <rect x="-75" y="-11" width="150" height="22" rx="11" />
                              <text y="4" textAnchor="middle">◎ {t("camera.landmarkHead")}</text>
                            </g>

                            {/* Hump / Dorsal Ridge */}
                            <path
                              className="hud-hump-guide"
                              d={
                                isBuffaloGuide
                                  ? `M ${humpX - 60} 175 Q ${humpX} 148 ${humpX + 60} 175`
                                  : `M ${humpX - 55} 175 Q ${humpX} 105 ${humpX + 55} 175`
                              }
                            />
                            <circle className="hud-anchor" cx={humpX} cy="138" r="4" />
                            <g className="hud-label" transform={`translate(${humpX}, ${humpY - 35})`}>
                              <rect x={-guideWidth / 2} y="-11" width={guideWidth} height="22" rx="11" />
                              <text y="4" textAnchor="middle">{guideLabel}</text>
                            </g>

                            {/* Dewlap */}
                            <path
                              className="hud-dewlap-guide"
                              d={`M ${dewlapX - 28} ${dewlapY - 35} Q ${dewlapX + 12} ${dewlapY} ${dewlapX - 8} ${dewlapY + 40}`}
                            />
                            <circle className="hud-anchor" cx={dewlapX} cy={dewlapY} r="4" />
                            <g className="hud-label" transform={`translate(${dewlapX}, ${dewlapY + 60})`}>
                              <rect x="-68" y="-11" width="136" height="22" rx="11" />
                              <text y="4" textAnchor="middle">◡ {t("camera.landmarkDewlap")}</text>
                            </g>

                            {/* Tail / Rump */}
                            <path
                              className="hud-tail-guide"
                              d={`M ${tailX} 175 Q ${tailX + (facingRight ? 18 : -18)} 255 ${tailX} 345`}
                            />
                            <circle className="hud-anchor" cx={tailX} cy={tailY} r="4" />
                            <g className="hud-label" transform={`translate(${tailX}, 365)`}>
                              <rect x="-58" y="-11" width="116" height="22" rx="11" />
                              <text y="4" textAnchor="middle">⫶ Tail / Rump</text>
                            </g>
                          </g>
                        )}
                      </svg>

                      <div className="camera-status-pills">
                        <span className={`camera-status-pill status-${lightingStatus.key}`}>
                          <i aria-hidden="true" /> {t("camera.lighting")}: {lightingStatus.label}
                        </span>
                        <span className={`camera-status-pill status-${framingStatus.key}`}>
                          <i aria-hidden="true" /> {t("camera.animalInFrame")}: {framingStatus.label}
                        </span>
                        <span className={`camera-status-pill status-${sharpnessStatus.key}`}>
                          <i aria-hidden="true" /> {t("camera.sharpness")}: {sharpnessStatus.label}
                        </span>
                      </div>
                    </div>
                  )}

                  <div className="camera-guidance">
                    <QualityPill quality={quality} />
                    <div className="camera-hud-right">
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
                      <span className="camera-zero-crop-badge">{t("camera.zeroCropBadge")}</span>
                    </div>
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
                <select
                  className={`camera-toggle-btn camera-silhouette-select ${silhouette !== "off" ? "is-on" : ""}`}
                  value={silhouette}
                  onChange={(event) => setSilhouette(event.target.value)}
                  title={t("camera.silhouette")}
                  aria-label={t("camera.silhouette")}
                >
                  <option value="off">◫ {t("camera.silhouetteOff")}</option>
                  <option value="cattle">◎ {t("camera.silhouetteCattle")}</option>
                  <option value="buffalo">◎ {t("camera.silhouetteBuffalo")}</option>
                </select>
                {silhouette !== "off" && (
                  <button
                    type="button"
                    className="camera-toggle-btn camera-facing-btn"
                    onClick={() => setSilhouetteFlipped((value) => !value)}
                    title={t("camera.flipSilhouette")}
                  >
                    {facingRight ? t("camera.facingRight") : t("camera.facingLeft")}
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
