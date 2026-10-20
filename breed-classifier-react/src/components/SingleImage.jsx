import { useEffect, useRef, useState } from "react";
import CameraCapture from "./CameraCapture.jsx";
import ModelSelect from "./ModelSelect.jsx";
import ResultsPanel from "./ResultsPanel.jsx";
import PredictionProgress from "./PredictionProgress.jsx";
import { predict, exportOdt, downloadBlob } from "../api.js";
import {
  formatBytes,
  optimizeImageTo260,
  releaseOptimizerResult,
} from "../utils/imageOptimizer.js";

const ACCEPT = ".png,.jpg,.jpeg,.bmp,.webp";

export default function SingleImage({ models, model, setModel, devMode, predictionEvent }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [sourceType, setSourceType] = useState(null); // 'camera' | 'upload' | 'sample'
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [optimizerMeta, setOptimizerMeta] = useState(null);
  const inputRef = useRef(null);
  const resultRef = useRef(null);
  const requestRef = useRef(0);

  useEffect(
    () => () => {
      if (resultRef.current) {
        releaseOptimizerResult(resultRef.current);
        resultRef.current = null;
      }
    },
    []
  );

  const pickFile = async (f, source = "upload") => {
    if (!f || !f.type.startsWith("image/")) return;
    const requestId = ++requestRef.current;
    setOptimizing(true);
    try {
      const optimized = await optimizeImageTo260(f, {
        fitMode: "pad",
        padStyle: "imagenet_mean",
        allowUpscale: false,
      });
      if (requestId !== requestRef.current) {
        releaseOptimizerResult(optimized);
        return;
      }
      if (resultRef.current) releaseOptimizerResult(resultRef.current);
      resultRef.current = optimized;
      setFile(optimized.file);
      setSourceType(source);
      setPreview(optimized.dataUrl);
      setOptimizerMeta(optimized.meta);
      setResult(null);
    } catch (e) {
      setResult({ error: `Image optimization failed: ${e.message}` });
    } finally {
      setOptimizing(false);
    }
  };

  const clear = () => {
    requestRef.current += 1;
    setOptimizing(false);
    if (resultRef.current) {
      releaseOptimizerResult(resultRef.current);
      resultRef.current = null;
    }
    setFile(null);
    setSourceType(null);
    setPreview(null);
    setOptimizerMeta(null);
    setResult(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const analyze = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const data = await predict(file, model, file.name);
      setResult(data);
    } catch (e) {
      setResult({ error: "Prediction failed: " + e.message });
    } finally {
      setBusy(false);
    }
  };

  const doExport = async () => {
    setExporting(true);
    try {
      const blob = await exportOdt(result, "single");
      downloadBlob(blob, `breed_test_result_${Date.now()}.odt`);
    } catch (e) {
      alert(e.message);
    } finally {
      setExporting(false);
    }
  };

  // Helper to load a demo sample image for quick testing
  const loadSample = async (type) => {
    const canvas = document.createElement("canvas");
    canvas.width = 600;
    canvas.height = 450;
    const ctx = canvas.getContext("2d");

    if (type === "gir") {
      // Warm terracotta pasture demo with cattle silhouette
      const grad = ctx.createLinearGradient(0, 0, 0, 450);
      grad.addColorStop(0, "#d97706");
      grad.addColorStop(0.6, "#78350f");
      grad.addColorStop(1, "#292524");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 600, 450);

      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 28px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("🐄 Sample: Gir Cattle (Demo)", 300, 200);
      ctx.font = "16px sans-serif";
      ctx.fillStyle = "#fef08a";
      ctx.fillText("Bos indicus — Famous dairy breed of Gujarat", 300, 240);
    } else {
      // Rich dark green pasture demo with buffalo silhouette
      const grad = ctx.createLinearGradient(0, 0, 0, 450);
      grad.addColorStop(0, "#065f46");
      grad.addColorStop(0.6, "#064e3b");
      grad.addColorStop(1, "#022c22");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 600, 450);

      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 28px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("🐃 Sample: Murrah Buffalo (Demo)", 300, 200);
      ctx.font = "16px sans-serif";
      ctx.fillStyle = "#a7f3d0";
      ctx.fillText("Bubalus bubalis — Premier dairy buffalo breed", 300, 240);
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const demoFile = new File([blob], `sample_${type}_demo.jpg`, { type: "image/jpeg" });
        pickFile(demoFile, "sample");
      }
    }, "image/jpeg", 0.95);
  };

  const selectedMeta = models.find((m) => m.name === model);

  return (
    <div className="container">
      {/* Input Card */}
      <div className="card glass-card">
        <div className="card-header-row">
          <h2>📸 Select or Capture Image</h2>
        </div>

        {!preview ? (
          <>
            {/* Primary Action Buttons: Take Photo & Upload */}
            <div className="input-action-grid">
              <button
                type="button"
                className="action-tile camera-tile"
                onClick={() => setCameraOpen(true)}
                disabled={busy || optimizing}
              >
                <div className="tile-icon-wrap">
                  <span className="tile-icon">📷</span>
                  <span className="live-badge">LIVE</span>
                </div>
                <div className="tile-content">
                  <span className="tile-title">Take Photo Live</span>
                  <span className="tile-sub">Use your webcam or phone camera</span>
                </div>
              </button>

              <button
                type="button"
                className="action-tile upload-tile"
                onClick={() => inputRef.current?.click()}
                disabled={busy || optimizing}
              >
                <div className="tile-icon-wrap">
                  <span className="tile-icon">📁</span>
                </div>
                <div className="tile-content">
                  <span className="tile-title">Upload Photo</span>
                  <span className="tile-sub">Browse from device or computer</span>
                </div>
              </button>
            </div>

            {/* Drag & Drop Zone */}
            <div
              className={`dropzone modern-dropzone ${dragOver ? "drag-over" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                if (e.dataTransfer.files.length) {
                  pickFile(e.dataTransfer.files[0], "upload");
                }
              }}
            >
              <div className="dropzone-inner">
                <span className="drop-icon">☁️</span>
                <p className="drop-text">
                  <strong>Drag & drop an image here</strong> or click to browse
                </p>
                <div className="formats-pill">PNG · JPG · JPEG · WEBP · BMP</div>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPT}
                hidden
                onChange={(e) => {
                  if (e.target.files.length) {
                    pickFile(e.target.files[0], "upload");
                  }
                }}
              />
            </div>

            {/* Quick Demo Samples */}
            <div className="quick-samples-bar">
              <span className="samples-label">Quick Test:</span>
              <button
                type="button"
                className="sample-chip"
                onClick={() => loadSample("gir")}
                disabled={busy || optimizing}
              >
                🐄 Gir Cattle Sample
              </button>
              <button
                type="button"
                className="sample-chip"
                onClick={() => loadSample("murrah")}
                disabled={busy || optimizing}
              >
                🐃 Murrah Buffalo Sample
              </button>
            </div>
          </>
        ) : (
          /* Preview state when an image is ready */
          <div className="preview-section">
            <div className="preview-top-bar">
              <span className={`source-pill ${sourceType}`}>
                {sourceType === "camera" && "📷 Live Photo"}
                {sourceType === "upload" && "📁 Uploaded File"}
                {sourceType === "sample" && "🧪 Demo Sample"}
              </span>
              <span className="file-size-pill">
                {(file.size / 1024).toFixed(0)} KB
              </span>
            </div>

            {optimizerMeta && (
              <div className="optimizer-bar optimizer-bar-compact">
                <span className="optimizer-badge">
                  {optimizerMeta.originalWidth}×{optimizerMeta.originalHeight} → 260×260 •{" "}
                  {optimizerMeta.fitMode === "stretch" ? "Stretch" : "Zero Crop"} •{" "}
                  {formatBytes(optimizerMeta.optimizedBytes)}
                </span>
              </div>
            )}

            <div className="preview-container modern-preview preview-260-container">
              <img src={preview} alt="Selected livestock preview" />
              <button
                className="clear-btn"
                title="Remove image"
                onClick={clear}
                aria-label="Remove image"
              >
                ✕
              </button>
            </div>

            <div className="filename-label">
              <span className="fname-label-text">File:</span>
              <span className="fname">{file.name}</span>
            </div>

            {/* Change / Retake options */}
            <div className="change-photo-actions">
              <button
                type="button"
                className="btn ghost btn-sm"
                onClick={() => setCameraOpen(true)}
                disabled={busy || optimizing}
              >
                📷 Retake with Camera
              </button>
              <button
                type="button"
                className="btn ghost btn-sm"
                onClick={() => inputRef.current?.click()}
                disabled={busy || optimizing}
              >
                📁 Choose Different File
              </button>
            </div>
          </div>
        )}

        {/* Model Selection */}
        <div className="model-select-wrapper">
          <ModelSelect
            models={models}
            value={model}
            onChange={setModel}
            meta={selectedMeta}
          />
        </div>

        {/* Main Analyze Button */}
        <button
          className={`btn primary full analyze-cta-btn ${busy || optimizing ? "busy" : ""}`}
          onClick={analyze}
          disabled={!file || busy || optimizing}
        >
          {optimizing ? (
            <span className="btn-loading-flex">
              <span className="spinner-icon" />
              <span>Optimizing to 260×260…</span>
            </span>
          ) : busy ? (
            <span className="btn-loading-flex">
              <span className="spinner-icon" />
              <span>Analyzing Livestock Breed…</span>
            </span>
          ) : (
            <span>✨ Analyze Breed Now</span>
          )}
        </button>
        
        <PredictionProgress predictionEvent={predictionEvent} isBusy={busy} />
      </div>

      {/* Results Card */}
      <div className="card glass-card">
        <div className="card-header-row">
          <h2>📊 Prediction Results</h2>
        </div>
        <ResultsPanel
          result={result}
          devMode={devMode}
          filename={file?.name}
          onExport={doExport}
          exporting={exporting}
        />
      </div>

      {/* Live Camera Modal */}
      {cameraOpen && (
        <CameraCapture
          onClose={() => setCameraOpen(false)}
          onCapture={(capturedFile, capturedDataUrl, capturedMeta) => {
            if (!capturedFile) return;
            requestRef.current += 1;
            setOptimizing(false);
            if (resultRef.current) releaseOptimizerResult(resultRef.current);
            resultRef.current = null;
            setFile(capturedFile);
            setSourceType("camera");
            setPreview(capturedDataUrl);
            setOptimizerMeta(capturedMeta || null);
            setResult(null);
          }}
        />
      )}
    </div>
  );
}
