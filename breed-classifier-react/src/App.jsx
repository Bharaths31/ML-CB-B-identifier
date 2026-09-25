import { useEffect, useState } from "react";
import SingleImage from "./components/SingleImage.jsx";
import Batch from "./components/Batch.jsx";
import DevTools from "./components/DevTools.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";
import { getModels } from "./api.js";
import { useSocket } from "./hooks/useSocket.js";

const LS_KEY = "breed-dev-mode";

export default function App() {
  const [devMode, setDevMode] = useState(() => localStorage.getItem(LS_KEY) === "1");
  const [tab, setTab] = useState("single");
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [device, setDevice] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [loadingModels, setLoadingModels] = useState(true);

  const { connected, serverStatus, predictionEvent } = useSocket();

  useEffect(() => {
    localStorage.setItem(LS_KEY, devMode ? "1" : "0");
  }, [devMode]);

  const loadModels = () => {
    setLoadingModels(true);
    setApiError(null);
    getModels()
      .then((data) => {
        setModels(data.models || []);
        // We get device from serverStatus now, but keep this fallback
        setDevice(serverStatus?.device || data.device);
        const preferred =
          data.models?.find((m) => m.name.includes("phase2"))?.name ||
          data.models?.[0]?.name ||
          "";
        setModel(preferred);
      })
      .catch((e) => {
        setApiError(
          "Backend connection issue: Could not reach model server at http://localhost:8501 (" +
            e.message +
            "). Run `python test_model.py` to start the backend."
        );
      })
      .finally(() => {
        setLoadingModels(false);
      });
  };

  useEffect(() => {
    loadModels();
  }, []);

  const tabs = [
    ["single", "📸 Single Image"],
    ["batch", "📁 Batch Analysis"],
    ...(devMode ? [["dev", "🛠️ Dev Tools"]] : []),
  ];

  return (
    <div className="app-layout">
      {/* Top Navigation Bar */}
      <header className="top-nav">
        <div className="nav-brand">
          <span className="brand-logo-icon">🐄</span>
          <div className="brand-text">
            <span className="brand-title">Breed Classifier</span>
            <span className="brand-tagline">Livestock AI Vision</span>
          </div>
        </div>

        <div className="nav-actions">
          {/* Theme switcher button */}
          <ThemeToggle />

          {/* Dev mode switch */}
          <label className="mode-toggle-pill" title="Toggle developer mode">
            <input
              type="checkbox"
              checked={devMode}
              onChange={(e) => {
                setDevMode(e.target.checked);
                if (!e.target.checked && tab === "dev") setTab("single");
              }}
            />
            <span className="mode-slider" />
            <span className="mode-label">{devMode ? "Dev" : "User"}</span>
          </label>
        </div>
      </header>

      {/* Main Hero Header */}
      <div className="hero-banner">
        <div className="hero-badge">
          <span className="pulse-indicator" />
          <span>AI-Powered Livestock Identification</span>
        </div>
        <h1>Identify Cattle & Buffalo Breeds</h1>
        <p>
          Take a live photo or upload an image to identify Indian & global breeds
          with deep learning confidence scores.
        </p>
      </div>

      {/* Status Indicators Bar */}
      <div className="status-bar">
        <span className={`chip ${connected ? "online" : "offline"}`} style={{
            backgroundColor: connected ? "var(--green)" : "var(--red)",
            color: "var(--background)",
            fontWeight: "bold"
        }}>
          {connected ? "🟢 Live Connected" : "🔴 Disconnected"}
        </span>

        {serverStatus?.device ? (
          <span className={"chip " + (serverStatus.device === "cuda" ? "gpu" : "cpu")}>
            {serverStatus.device === "cuda" ? "🟢 GPU (CUDA Active)" : "🟡 CPU Execution"}
          </span>
        ) : device ? (
          <span className={"chip " + (device === "cuda" ? "gpu" : "cpu")}>
            {device === "cuda" ? "🟢 GPU (CUDA Active)" : "🟡 CPU Execution"}
          </span>
        ) : null}

        <span className="chip models-count-chip">
          📦 {serverStatus ? serverStatus.models_count : models.length} Models Loaded
        </span>

        <span className={"chip " + (devMode ? "dev" : "user")}>
          {devMode ? "🛠️ Developer Mode" : "👤 Standard Mode"}
        </span>
      </div>

      {/* Backend Alert Banner if disconnected */}
      {apiError && (
        <div className="backend-alert-banner">
          <span className="alert-icon">⚠️</span>
          <div className="alert-content">
            <strong>Backend Unreachable:</strong> {apiError}
          </div>
          <button className="btn ghost btn-sm alert-retry-btn" onClick={loadModels}>
            🔄 Retry
          </button>
        </div>
      )}

      {/* Tab Navigation */}
      <nav className="tabs" role="tablist">
        {tabs.map(([id, label]) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            className={"tab-btn" + (tab === id ? " active" : "")}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      {/* Active Tab View */}
      <main className="tab-content">
        {tab === "single" && (
          <SingleImage
            models={models}
            model={model}
            setModel={setModel}
            devMode={devMode}
            predictionEvent={predictionEvent}
          />
        )}
        {tab === "batch" && (
          <Batch
            models={models}
            model={model}
            setModel={setModel}
            devMode={devMode}
            predictionEvent={predictionEvent}
          />
        )}
        {tab === "dev" && devMode && <DevTools models={models} />}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>Breed Classifier AI · Supporting Cattle & Buffalo Breed Classification</p>
      </footer>
    </div>
  );
}
