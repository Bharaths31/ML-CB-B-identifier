import { useEffect, useRef, useState } from "react";
import ImageCropper from "./ImageCropper.jsx";
import { useI18n } from "../i18n/I18nProvider.jsx";
import {
  formatBytes,
  optimizeImageTo260,
  releaseOptimizerResult,
} from "../utils/imageOptimizer.js";

const DEFAULT_FIT = { fitMode: "pad", padStyle: "imagenet_mean" };

export default function ImageUpload({
  onImageSelected,
  currentPreview,
  optimizerMeta,
  onClear,
}) {
  const { t } = useI18n();
  const [isDragging, setIsDragging] = useState(false);
  const [result, setResult] = useState(null);
  const [activeFile, setActiveFile] = useState(null);
  const [optimizing, setOptimizing] = useState(false);
  const [error, setError] = useState(null);
  const [showOriginal, setShowOriginal] = useState(false);
  const [fit, setFit] = useState(DEFAULT_FIT);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(null);
  const [manuallyCropped, setManuallyCropped] = useState(false);

  const fileInputRef = useRef(null);
  const requestRef = useRef(0);
  const resultRef = useRef(null);

  // A new parent preview from the camera replaces any local upload state.
  useEffect(() => {
    if (currentPreview && !resultRef.current) {
      setShowOriginal(false);
    }
  }, [currentPreview]);

  // Parent cleared the image (ResultCard "Classify another" / Remove).
  useEffect(() => {
    if (!currentPreview && resultRef.current) {
      releaseOptimizerResult(resultRef.current);
      resultRef.current = null;
      setResult(null);
      setActiveFile(null);
      setShowOriginal(false);
    }
  }, [currentPreview]);

  useEffect(
    () => () => {
      requestRef.current += 1;
      if (resultRef.current) {
        releaseOptimizerResult(resultRef.current);
        resultRef.current = null;
      }
    },
    []
  );

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };
  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) processFile(file);
  };
  const handleFileInput = (event) => {
    const file = event.target.files?.[0];
    if (file) processFile(file);
    event.target.value = "";
  };

  const optimize = async (file, nextFit, { cropped = false } = {}) => {
    const requestId = ++requestRef.current;
    setOptimizing(true);
    setError(null);
    try {
      const next = await optimizeImageTo260(file, {
        ...nextFit,
        allowUpscale: false,
      });
      if (requestId !== requestRef.current) {
        releaseOptimizerResult(next);
        return null;
      }
      if (cropped) next.meta.manuallyCropped = true;
      if (resultRef.current) releaseOptimizerResult(resultRef.current);
      resultRef.current = next;
      setResult(next);
      setActiveFile(file);
      setManuallyCropped(cropped);
      setShowOriginal(false);
      setOptimizing(false);
      onImageSelected?.(next.file, next.dataUrl, next.meta, next);
      return next;
    } catch (err) {
      if (requestId === requestRef.current) {
        setOptimizing(false);
        setError(err?.message || t("upload.invalid"));
      }
      return null;
    }
  };

  const processFile = async (file) => {
    if (!file.type.startsWith("image/")) {
      alert(t("upload.invalid"));
      return;
    }
    setEditing(false);
    setDraft(null);
    await optimize(file, fit);
  };

  const handleConfirmCrop = async (file) => {
    setEditing(false);
    setDraft(null);
    await optimize(file, fit, { cropped: true });
  };

  const openAdjust = () => {
    if (!resultRef.current?.originalDataUrl) return;
    setDraft({
      url: resultRef.current.originalDataUrl,
      name: activeFile?.name || "image.jpg",
    });
    setEditing(true);
  };

  const setMode = (mode, padStyle = fit.padStyle) => {
    const next = { fitMode: mode, padStyle };
    setFit(next);
    if (activeFile) optimize(activeFile, next, { cropped: manuallyCropped });
  };

  const handleRemove = () => {
    requestRef.current += 1;
    setOptimizing(false);
    if (resultRef.current) {
      releaseOptimizerResult(resultRef.current);
      resultRef.current = null;
    }
    setResult(null);
    setActiveFile(null);
    setFit(DEFAULT_FIT);
    setManuallyCropped(false);
    setShowOriginal(false);
    onClear?.();
  };

  if (editing && draft) {
    return (
      <div className="upload-container upload-editing">
        <ImageCropper
          imageUrl={draft.url}
          fileName={draft.name}
          onConfirm={handleConfirmCrop}
          onCancel={() => {
            setEditing(false);
            setDraft(null);
          }}
          confirmLabel={t("optimizer.keep")}
        />
      </div>
    );
  }

  const previewSrc = result
    ? showOriginal
      ? result.originalDataUrl
      : result.dataUrl
    : currentPreview;
  const meta = result?.meta || optimizerMeta || null;
  const canReoptimize = Boolean(result && activeFile);

  return (
    <div className="upload-container">
      {optimizing && !result ? (
        <div className="optimizer-loading-card">
          <span className="camera-spinner" />
          <p>{t("optimizer.optimizing")}</p>
        </div>
      ) : previewSrc ? (
        <div className="preview-card preview-card-square">
          <div className="preview-260-frame">
            <img src={previewSrc} alt="Optimized 260 by 260 model input" />
            <span className="preview-260-badge">
              {showOriginal ? t("optimizer.original") : "260×260"}
            </span>
            {optimizing && (
              <div className="optimizer-busy-overlay">
                <span className="camera-spinner" />
                <span>{t("optimizer.optimizing")}</span>
              </div>
            )}
          </div>

          {meta && (
            <div className="optimizer-bar">
              <span className="optimizer-badge">
                {t("optimizer.original")}: {meta.originalWidth}×{meta.originalHeight}
                {meta.originalAspectRatio ? ` (${meta.originalAspectRatio})` : ""}
                {meta.originalBytes ? ` · ${formatBytes(meta.originalBytes)}` : ""} → 260×260
                {` (${formatBytes(meta.optimizedBytes)}`}
                {meta.savingsPct
                  ? ` • ${meta.savingsPct > 0 ? "-" : "+"}${Math.abs(meta.savingsPct)}%`
                  : ""}
                {") • "}
                {meta.manuallyCropped || manuallyCropped
                  ? t("camera.manualCrop")
                  : t("optimizer.zeroCrop")}
                {meta.fitMode === "stretch" ? ` • ${t("optimizer.stretch")}` : ""}
                {meta.upscaled ? ` • ${t("optimizer.upscaled")}` : ""}
              </span>
              <div className="optimizer-preview-toggle">
                <button
                  type="button"
                  className={`optimizer-toggle-btn ${!showOriginal ? "is-on" : ""}`}
                  onClick={() => setShowOriginal(false)}
                  disabled={!result}
                >
                  {t("optimizer.viewModel")}
                </button>
                <button
                  type="button"
                  className={`optimizer-toggle-btn ${showOriginal ? "is-on" : ""}`}
                  onClick={() => setShowOriginal(true)}
                  disabled={!result?.originalDataUrl}
                >
                  {t("optimizer.viewOriginal")}
                </button>
              </div>
            </div>
          )}

          {meta && (
            <div className="optimizer-meta-grid">
              <div>
                <span>{t("optimizer.original")}</span>
                <strong>
                  {meta.originalWidth}×{meta.originalHeight}
                </strong>
              </div>
              <div>
                <span>{t("optimizer.modelInput")}</span>
                <strong>260×260</strong>
              </div>
              <div>
                <span>{t("optimizer.scaled")}</span>
                <strong>
                  {meta.scaledInnerWidth}×{meta.scaledInnerHeight}
                </strong>
              </div>
              <div>
                <span>{t("optimizer.padding")}</span>
                <strong>
                  {meta.padX}, {meta.padY}
                </strong>
              </div>
              <div>
                <span>{t("optimizer.bytes")}</span>
                <strong>{formatBytes(meta.optimizedBytes)}</strong>
              </div>
              <div>
                <span>{t("optimizer.time")}</span>
                <strong>{meta.durationMs} ms</strong>
              </div>
            </div>
          )}

          {canReoptimize && (
            <div className="optimizer-controls" role="group" aria-label={t("optimizer.fitMode")}>
              <button
                type="button"
                className={`optimizer-mode-btn ${
                  fit.fitMode === "pad" && fit.padStyle === "imagenet_mean" ? "is-on" : ""
                }`}
                onClick={() => setMode("pad", "imagenet_mean")}
                disabled={optimizing}
              >
                {t("optimizer.padMean")}
              </button>
              <button
                type="button"
                className={`optimizer-mode-btn ${
                  fit.fitMode === "pad" && fit.padStyle === "ambient_blur" ? "is-on" : ""
                }`}
                onClick={() => setMode("pad", "ambient_blur")}
                disabled={optimizing}
              >
                {t("optimizer.padBlur")}
              </button>
              <button
                type="button"
                className={`optimizer-mode-btn ${
                  fit.fitMode === "pad" && fit.padStyle === "black" ? "is-on" : ""
                }`}
                onClick={() => setMode("pad", "black")}
                disabled={optimizing}
              >
                {t("optimizer.padBlack")}
              </button>
              <button
                type="button"
                className={`optimizer-mode-btn ${fit.fitMode === "stretch" ? "is-on" : ""}`}
                onClick={() => setMode("stretch")}
                disabled={optimizing}
              >
                {t("optimizer.stretch")}
              </button>
            </div>
          )}

          {error && <p className="camera-error-inline">{error}</p>}

          <div className="preview-actions">
            {canReoptimize && (
              <button type="button" className="btn btn-outline" onClick={openAdjust}>
                {t("optimizer.adjust")}
              </button>
            )}
            <button type="button" className="btn btn-secondary" onClick={handleRemove}>
              {t("actions.remove")}
            </button>
            <button type="button" className="btn btn-outline" onClick={() => fileInputRef.current?.click()}>
              {t("actions.chooseAnother")}
            </button>
          </div>
        </div>
      ) : (
        <div
          className={`dropzone ${isDragging ? "dropzone-active" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") fileInputRef.current?.click();
          }}
        >
          <div className="dropzone-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
              <path d="M12 12v9" />
              <path d="m16 16-4-4-4 4" />
            </svg>
          </div>
          <div className="dropzone-text">
            <h3 className="dropzone-title">{t("upload.dropzoneTitle")}</h3>
            <p className="dropzone-subtitle">
              {t("upload.dropzoneSubtitle", { browse: "" }).trimEnd()}{" "}
              <span className="highlight-text">{t("upload.browse")}</span>
            </p>
            <p className="dropzone-hint">{t("upload.hint")}</p>
          </div>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={handleFileInput}
      />
    </div>
  );
}
