import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import ImageUpload from "../components/ImageUpload.jsx";
import CameraCapture from "../components/CameraCapture.jsx";
import ResultCard from "../components/ResultCard.jsx";
import ThemeToggle from "../components/ThemeToggle.jsx";
import LanguageToggle from "../components/LanguageToggle.jsx";
import HerdSessionDrawer from "../components/HerdSessionDrawer.jsx";
import AdvancedPanel from "../components/AdvancedPanel.jsx";
import { predict } from "../api.js";
import { useSocket } from "../hooks/useSocket.js";
import { useSharedConfig } from "../hooks/useSharedConfig.js";
import { useImageOptimizer } from "../hooks/useImageOptimizer.js";
import { useI18n } from "../i18n/I18nProvider.jsx";
import {
  addSessionEntry,
  makeSessionEntry,
  readSession,
  sessionStats,
  thumbnailFromDataUrl,
} from "../utils/sessionStore.js";

export default function UserPage() {
  const { t } = useI18n();
  // `?mode=camera` is a deep link used by QR codes / kiosk links.
  const [inputMode, setInputMode] = useState(() => {
    try {
      return new URLSearchParams(window.location.search).get("mode") === "camera" ? "camera" : "upload";
    } catch {
      return "upload";
    }
  });
  const optimizer = useImageOptimizer();
  const selectedFile = optimizer.result?.file || null;
  const previewUrl = optimizer.original?.dataUrl || null;
  const optimizerMeta = optimizer.result?.meta || null;

  const [quality, setQuality] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [session, setSession] = useState(() => readSession());
  const [sessionOpen, setSessionOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const { connected, serverStatus, predictionEvent, socket } = useSocket();
  const { config } = useSharedConfig(socket);

  // A new model input (new upload, crop, or fit change) invalidates the
  // previous prediction. This keeps the explicit classify action visible.
  useEffect(() => {
    setPredictionResult(null);
    setError(null);
  }, [optimizer.result?.file]);

  const handleImageSelected = (file) => {
    setPredictionResult(null);
    setError(null);
    optimizer.loadFile(file);
  };

  const handleApplyCrop = async (file) => {
    // Clear the old result immediately, before the new optimization finishes.
    setPredictionResult(null);
    setError(null);
    return optimizer.applyCrop(file);
  };

  // CameraCapture passes (File, 260x260 dataUrl, optimizer meta, quality report).
  const handleCameraCapture = (file, dataUrl, meta, report) => {
    optimizer.adoptResult(file, dataUrl, meta);
    setQuality(report);
    setPredictionResult(null);
    setError(null);
    setInputMode("upload");
  };

  const handleClear = () => {
    optimizer.reset();
    setQuality(null);
    setPredictionResult(null);
    setError(null);
  };

  const handleRetake = () => {
    handleClear();
    setInputMode("camera");
  };

  const activeModelId = config?.active_model || serverStatus?.active_model || "";
  const activeModelDisplay = activeModelId || t("header.defaultModel");

  const handleRunPrediction = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setError(null);

    try {
      // Always log/export the original upload name, even though the file sent
      // to the model is the optimized 260x260 copy.
      const filename =
        optimizer.original?.name || selectedFile.name || "capture.jpg";
      const res = await predict(selectedFile, activeModelId, filename);

      if (res.error) throw new Error(res.error);
      setPredictionResult(res);

      // Log the confirmed prediction in the herd session (thumbnail kept small).
      const thumbnail = await thumbnailFromDataUrl(previewUrl);
      const entry = makeSessionEntry({ result: res, quality, thumbnail, filename });
      setSession((prev) => addSessionEntry(entry, prev));
    } catch (err) {
      setError(err.message || "Prediction failed. Check backend connection.");
    } finally {
      setAnalyzing(false);
    }
  };

  const stats = sessionStats(session);

  return (
    <div className="user-page-container">
      <header className="site-header">
        <div className="header-brand">
          <div className="brand-icon-wrapper">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2a4 4 0 0 1 4 4c0 .88-.29 1.7-.78 2.36l.28.28a6 6 0 0 1 1.76 4.24V17a3 3 0 0 1-3 3H9.74a3 3 0 0 1-3-3v-4.12a6 6 0 0 1 1.76-4.24l.28-.28A3.99 3.99 0 0 1 8 6a4 4 0 0 1 4-4Z" />
            </svg>
          </div>
          <div className="brand-info">
            <h1 className="brand-title">{t("app.title")}</h1>
            <span className="brand-subtitle">{t("app.subtitle")}</span>
          </div>
        </div>

        <div className="header-nav-right">
          <div className="status-badge" title={connected ? t("header.connected") : t("header.reconnecting")}>
            <span className={`status-dot ${connected ? "status-online" : "status-offline"}`} />
            <span className="status-text">{connected ? t("header.connected") : t("header.reconnecting")}</span>
          </div>

          <button
            type="button"
            className="advanced-trigger"
            onClick={() => setAdvancedOpen(true)}
            title={t("advanced.title")}
          >
            ⚙ {t("advanced.open")}
          </button>

          <div className="model-chip" title={`${t("header.model")}: ${activeModelDisplay}`}>
            <span className="chip-label">{t("header.model")}:</span>
            <span className="chip-value">{activeModelDisplay}</span>
          </div>

          <button
            type="button"
            className="session-trigger"
            onClick={() => setSessionOpen(true)}
            title={t("actions.openSession")}
          >
            <span className="session-trigger-icon" aria-hidden="true">▤</span>
            <span className="session-trigger-label">{t("session.tab")}</span>
            <span className="session-trigger-count">{stats.total}</span>
          </button>

          <LanguageToggle />
          <ThemeToggle />

          <Link to="/dev" className="btn btn-outline dev-portal-btn">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
            {t("header.dev")}
          </Link>
        </div>
      </header>

      <main className="user-main-content">
        <section className="input-panel-section">
          <div className="card-box">
            <div className="card-tab-nav">
              <button
                type="button"
                className={`tab-btn ${inputMode === "upload" ? "tab-active" : ""}`}
                onClick={() => setInputMode("upload")}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                {t("tabs.upload")}
              </button>
              <button
                type="button"
                className={`tab-btn ${inputMode === "camera" ? "tab-active" : ""}`}
                onClick={() => setInputMode("camera")}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                  <circle cx="12" cy="13" r="4" />
                </svg>
                {t("tabs.camera")}
              </button>
            </div>

            <div className="tab-pane-content">
              {inputMode === "upload" && (
                <ImageUpload
                  original={optimizer.original}
                  busy={optimizer.busy}
                  error={optimizer.error}
                  framed={Boolean(optimizer.cropped)}
                  onFileSelected={handleImageSelected}
                  onApplyCrop={handleApplyCrop}
                  onClear={handleClear}
                  onClassify={handleRunPrediction}
                  analyzing={analyzing}
                  canClassify={Boolean(selectedFile) && !optimizer.busy && !analyzing}
                  showClassify={!predictionResult}
                />
              )}

              {inputMode === "camera" && (
                <div className="camera-tab-container">
                  <CameraCapture onCapture={handleCameraCapture} onClose={() => setInputMode("upload")} />
                </div>
              )}

              {analyzing && predictionEvent?.type === "progress" && (
                <div className="progress-status-box">
                  <div className="pulse-dot" />
                  <span>{predictionEvent.detail || t("actions.processing")}</span>
                </div>
              )}

              {error && (
                <div className="error-banner">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                  <span>{error}</span>
                </div>
              )}
            </div>
          </div>
        </section>

        {predictionResult && (
          <section className="results-panel-section">
            <ResultCard
              result={predictionResult}
              config={config}
              onReset={handleClear}
              onRetake={handleRetake}
              quality={quality}
              previewUrl={previewUrl}
              optimizerMeta={optimizerMeta}
            />
          </section>
        )}
      </main>

      <footer className="site-footer">
        <p>{t("footer.text")}</p>
      </footer>

      <HerdSessionDrawer
        open={sessionOpen}
        entries={session}
        onClose={() => setSessionOpen(false)}
        onCleared={() => setSession([])}
      />

      <AdvancedPanel
        open={advancedOpen}
        onClose={() => setAdvancedOpen(false)}
        optimizer={optimizer}
        model={activeModelId}
      />
    </div>
  );
}
