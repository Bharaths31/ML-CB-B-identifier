import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import ImageUpload from "../components/ImageUpload.jsx";
import CameraCapture from "../components/CameraCapture.jsx";
import ResultWorkspace from "../components/ResultWorkspace.jsx";
import PhotoGuideModal from "../components/PhotoGuideModal.jsx";
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
import { analyzeDataUrlQuality } from "../utils/cropAndQuality.js";
import {
  OUTBOX_EVENT,
  buildEscalationPayload,
  getBrowserGps,
  isPermanentEscalationError,
  outboxCount,
  enqueueEscalation,
  syncOutbox,
} from "../utils/escalationQueue.js";
import { createEscalation } from "../api.js";

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

  const [quality, setQuality] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [session, setSession] = useState(() => readSession());
  const [sessionOpen, setSessionOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [queuedEscalations, setQueuedEscalations] = useState(() => outboxCount());
  const [syncNotice, setSyncNotice] = useState(null);

  const { connected, serverStatus, predictionEvent, socket } = useSocket();
  const { config } = useSharedConfig(socket);

  // A new model input (new upload, crop, or fit change) invalidates the
  // previous prediction. This keeps the explicit classify action visible.
  useEffect(() => {
    setPredictionResult(null);
    setError(null);
  }, [optimizer.result?.file]);

  // The 260x260 model input is also the authoritative quality-gate canvas.
  // Recomputing after every upload/crop/camera frame removes the stale-camera
  // quality bug (the report needs subscores for the actual evaluated image).
  useEffect(() => {
    const dataUrl = optimizer.result?.dataUrl;
    if (!dataUrl) return undefined;
    let cancelled = false;
    analyzeDataUrlQuality(dataUrl)
      .then((report) => {
        if (!cancelled) setQuality(report);
      })
      .catch(() => {
        if (!cancelled) setQuality(null);
      });
    return () => {
      cancelled = true;
    };
  }, [optimizer.result?.dataUrl]);

  useEffect(() => {
    const refresh = () => setQueuedEscalations(outboxCount());
    window.addEventListener(OUTBOX_EVENT, refresh);
    return () => window.removeEventListener(OUTBOX_EVENT, refresh);
  }, []);

  const flushEscalations = async ({ silent = false } = {}) => {
    if (!outboxCount()) return;
    const result = await syncOutbox();
    setQueuedEscalations(result.remaining);
    if (result.dropped > 0) {
      setSyncNotice(t("escalation.dropped", { count: result.dropped }));
    } else if (!silent && result.synced > 0) {
      setSyncNotice(t("escalation.synced", { count: result.synced }));
    }
    if (result.dropped > 0 || (!silent && result.synced > 0)) {
      setTimeout(() => setSyncNotice(null), 5000);
    }
  };

  // Auto-sync once the /ws namespace reconnects (offline queue replay).
  useEffect(() => {
    if (connected) flushEscalations({ silent: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  const handleImageSelected = (file) => {
    setPredictionResult(null);
    setError(null);
    setQuality(null);
    optimizer.loadFile(file);
  };

  const handleApplyCrop = async (file, report = null) => {
    // Clear the old result and stale quality immediately, before the new
    // optimization finishes.
    setPredictionResult(null);
    setError(null);
    setQuality(report);
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
  const shortModel = (value) => {
    if (!value) return t("header.defaultModel");
    return value.length > 24 ? `${value.slice(0, 14)}…${value.slice(-4)}` : value;
  };
  const copyModel = () => {
    try {
      navigator.clipboard?.writeText(activeModelDisplay);
      setSyncNotice(t("header.modelCopied"));
      setTimeout(() => setSyncNotice(null), 2500);
    } catch {
      /* clipboard can be blocked; the tooltip still shows the full hash */
    }
  };

  const handleEscalate = async (resultToEscalate, { manual = false } = {}) => {
    const gps = await getBrowserGps();
    const payload = buildEscalationPayload({
      result: resultToEscalate || predictionResult,
      imageB64: optimizer.result?.dataUrl || previewUrl,
      quality,
      filename: optimizer.original?.name || selectedFile?.name || "capture.jpg",
      gps,
    });
    payload.manual_escalation = Boolean(manual);
    try {
      await createEscalation(payload);
      setSyncNotice(t("escalation.queuedOnline"));
    } catch (err) {
      if (isPermanentEscalationError(err)) {
        setError(err.message || t("escalation.rejected"));
      } else {
        enqueueEscalation(payload);
        setQueuedEscalations(outboxCount());
        setSyncNotice(t("escalation.queuedOffline"));
      }
    }
    setTimeout(() => setSyncNotice(null), 6000);
  };

  return (
    <div className="user-page-container">
      <header className="site-header">
        <div className="utility-bar">
          <div className="brand-zone">
            <div className="brand-emblem" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
                <path d="M12 2a4 4 0 0 1 4 4c0 .88-.29 1.7-.78 2.36l.28.28a6 6 0 0 1 1.76 4.24V17a3 3 0 0 1-3 3H9.74a3 3 0 0 1-3-3v-4.12a6 6 0 0 1 1.76-4.24l.28-.28A3.99 3.99 0 0 1 8 6a4 4 0 0 1 4-4Z" />
              </svg>
            </div>
            <div className="brand-text">
              <h1 className="brand-title">{t("app.title")}</h1>
              <span className="brand-divider" aria-hidden="true">|</span>
              <span className="brand-subtitle">{t("app.subtitle")}</span>
            </div>
          </div>

          <div className="telemetry-zone">
            <div className="system-pill" role="group" aria-label="System telemetry">
              <span className="telemetry-item" title={connected ? t("header.connected") : t("header.reconnecting")}>
                <span className={`telemetry-dot ${connected ? "online" : "offline"}`} />
                {connected ? t("header.connected") : t("header.reconnecting")}
              </span>
              <span className="telemetry-sep" aria-hidden="true" />
              <button
                type="button"
                className="telemetry-item telemetry-model"
                title={`${t("header.model")}: ${activeModelDisplay} — click to copy`}
                onClick={copyModel}
              >
                {shortModel(activeModelDisplay)}
              </button>
              <span className="telemetry-sep" aria-hidden="true" />
              <button
                type="button"
                className="telemetry-item"
                onClick={() => setSessionOpen(true)}
                title={t("actions.openSession")}
              >
                {t("session.tab")}: {stats.total}
              </button>
            </div>
          </div>

          <div className="utility-zone">
            {queuedEscalations > 0 && (
              <button
                type="button"
                className="escalation-outbox-badge"
                onClick={() => flushEscalations()}
                title={t("escalation.syncTitle")}
              >
                <span aria-hidden="true">⇅</span>
                {queuedEscalations}
              </button>
            )}

            {syncNotice && <span className="sync-notice">{syncNotice}</span>}

            <div className="nav-actions-group">
              <button
                type="button"
                className={`ghost-btn advanced-trigger ${advancedOpen ? "is-active" : ""}`}
                onClick={() => setAdvancedOpen(!advancedOpen)}
                title={t("advanced.title")}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <circle cx="12" cy="12" r="3" />
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                </svg>
                <span>{t("advanced.open")}</span>
              </button>

              <Link to="/dev" className="ghost-btn dev-portal-btn" title={t("header.dev")}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
                </svg>
                <span>{t("header.dev")}</span>
              </Link>
            </div>

            <span className="nav-group-sep" aria-hidden="true" />

            <div className="nav-prefs-group">
              <LanguageToggle />
              <ThemeToggle />
            </div>
          </div>

          <div className="mobile-header-actions">
            <span
              className={`mobile-status ${connected ? "is-online" : "is-offline"}`}
              title={connected ? t("header.connected") : t("header.reconnecting")}
            >
              <span className={`telemetry-dot ${connected ? "online" : "offline"}`} />
              {connected ? t("header.connected") : t("header.reconnecting")}
            </span>
            <button
              type="button"
              className="mobile-menu-trigger"
              aria-label={t("mobile.menu")}
              aria-expanded={mobileMenuOpen}
              onClick={() => setMobileMenuOpen(true)}
            >
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none"
                   stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                <path d="M4 7h16M4 12h16M4 17h16" />
              </svg>
            </button>
          </div>
        </div>
      </header>

      {mobileMenuOpen && (
        <div
          className="mobile-sheet-overlay"
          role="presentation"
          onClick={() => setMobileMenuOpen(false)}
        >
          <section
            className="mobile-sheet"
            role="dialog"
            aria-modal="true"
            aria-label={t("mobile.menu")}
            onClick={(event) => event.stopPropagation()}
          >
            <span className="mobile-sheet-handle" aria-hidden="true" />
            <header className="mobile-sheet-head">
              <div>
                <strong>{t("app.title")}</strong>
                <span>{shortModel(activeModelDisplay)}</span>
              </div>
              <button
                type="button"
                className="mobile-sheet-close"
                onClick={() => setMobileMenuOpen(false)}
                aria-label={t("mobile.close")}
              >
                ✕
              </button>
            </header>

            <div className="mobile-sheet-status">
              <span className={`telemetry-dot ${connected ? "online" : "offline"}`} />
              {connected ? t("header.connected") : t("header.reconnecting")}
              <span className="mobile-sheet-sep" aria-hidden="true">·</span>
              {t("session.tab")}: {stats.total}
            </div>

            <div className="mobile-sheet-actions">
              <button
                type="button"
                className="mobile-sheet-action"
                onClick={() => {
                  setMobileMenuOpen(false);
                  setAdvancedOpen(true);
                }}
              >
                ⚙ {t("advanced.open")}
              </button>
              <button
                type="button"
                className="mobile-sheet-action"
                onClick={() => {
                  setMobileMenuOpen(false);
                  setSessionOpen(true);
                }}
              >
                ▤ {t("session.tab")}
              </button>
              <Link
                to="/dev"
                className="mobile-sheet-action"
                onClick={() => setMobileMenuOpen(false)}
              >
                {t("header.dev")}
              </Link>
              {queuedEscalations > 0 && (
                <button
                  type="button"
                  className="mobile-sheet-action"
                  onClick={() => flushEscalations()}
                >
                  ⇅ {t("escalation.syncTitle")}
                </button>
              )}
            </div>

            <div className="mobile-sheet-controls">
              <LanguageToggle />
              <ThemeToggle />
            </div>
          </section>
        </div>
      )}

      <main className="user-main-content">
        <section className="input-panel-section">
          <div className="photo-guide-strip">
            <span className="photo-guide-strip-text">
              📸 {t("photoGuide.subtitle")}
            </span>
            <button
              type="button"
              className="btn btn-outline btn-sm"
              onClick={() => setGuideOpen(true)}
            >
              {t("photoGuide.button")}
            </button>
          </div>
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
                  quality={quality}
                  onFileSelected={handleImageSelected}
                  onApplyCrop={handleApplyCrop}
                  onClearCrop={optimizer.clearCrop}
                  onClear={handleClear}
                  onClassify={handleRunPrediction}
                  analyzing={analyzing}
                  canClassify={Boolean(selectedFile) && !optimizer.busy && !analyzing}
                  showClassify={!predictionResult}
                  modelInputUrl={optimizer.result?.dataUrl || null}
                  onQuality={setQuality}
                  onOpenGuide={() => setGuideOpen(true)}
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
          {!predictionResult && (
            <div className="spec-strip" aria-label="System specification">
              <div className="spec-item">
                <span className="spec-label">{t("spec.target")}</span>
                <span className="spec-value">{t("spec.value.target")}</span>
              </div>
              <div className="spec-item">
                <span className="spec-label">{t("spec.ood")}</span>
                <span className="spec-value">{t("spec.value.ood")}</span>
              </div>
              <div className="spec-item">
                <span className="spec-label">{t("spec.input")}</span>
                <span className="spec-value">{t("spec.value.input")}</span>
              </div>
            </div>
          )}
        </section>

        {predictionResult && (
          <section className="results-panel-section">
            <ResultWorkspace
              result={predictionResult}
              quality={quality}
              previewUrl={previewUrl}
              filename={optimizer.original?.name || selectedFile?.name}
              onReset={handleClear}
              onRetake={handleRetake}
              onOpenAdvanced={() => setAdvancedOpen(true)}
            />
          </section>
        )}
      </main>

      <footer className="site-footer status-bar">
        <span className="footer-specs">{t("footer.text")}</span>
        <span className="footer-meta">
          {shortModel(activeModelDisplay)} · {(serverStatus?.device || "cpu").toUpperCase()} · v2
        </span>
      </footer>

      <HerdSessionDrawer
        open={sessionOpen}
        entries={session}
        onClose={() => setSessionOpen(false)}
        onCleared={() => setSession([])}
      />

      <PhotoGuideModal open={guideOpen} onClose={() => setGuideOpen(false)} />

      <AdvancedPanel
        open={advancedOpen}
        onClose={() => setAdvancedOpen(false)}
        optimizer={optimizer}
        model={activeModelId}
        result={predictionResult}
        config={config}
        quality={quality}
        previewUrl={previewUrl}
        modelInputUrl={optimizer.result?.dataUrl || previewUrl}
        onEscalate={handleEscalate}
        onOpenExpert={() => {
          window.location.href = "/expert";
        }}
      />
    </div>
  );
}
