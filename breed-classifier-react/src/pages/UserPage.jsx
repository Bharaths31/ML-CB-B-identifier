import { useState } from "react";
import { Link } from "react-router-dom";
import ImageUpload from "../components/ImageUpload.jsx";
import CameraCapture from "../components/CameraCapture.jsx";
import ResultCard from "../components/ResultCard.jsx";
import ThemeToggle from "../components/ThemeToggle.jsx";
import { predict } from "../api.js";
import { useSocket } from "../hooks/useSocket.js";
import { useSharedConfig } from "../hooks/useSharedConfig.js";

export default function UserPage() {
  const [inputMode, setInputMode] = useState("upload"); // 'upload' | 'camera'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  const { connected, serverStatus, predictionEvent, socket } = useSocket();
  const { config } = useSharedConfig(socket);

  const handleImageSelected = (file, dataUrl) => {
    setSelectedFile(file);
    setPreviewUrl(dataUrl);
    setPredictionResult(null);
    setError(null);
  };

  const handleCameraCapture = (file, dataUrl) => {
    setSelectedFile(file);
    setPreviewUrl(dataUrl);
    setPredictionResult(null);
    setError(null);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPredictionResult(null);
    setError(null);
  };

  const handleRunPrediction = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setError(null);

    try {
      const activeModel = config?.active_model || serverStatus?.active_model || "";
      const filename = selectedFile.name || "capture.jpg";
      const res = await predict(selectedFile, activeModel, filename);

      if (res.error) {
        throw new Error(res.error);
      }
      setPredictionResult(res);
    } catch (err) {
      setError(err.message || "Prediction failed. Check backend connection.");
    } finally {
      setAnalyzing(false);
    }
  };

  const activeModelDisplay = config?.active_model || serverStatus?.active_model || "Default Model";

  return (
    <div className="user-page-container">
      {/* Top Navbar */}
      <header className="site-header">
        <div className="header-brand">
          <div className="brand-icon-wrapper">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2a4 4 0 0 1 4 4c0 .88-.29 1.7-.78 2.36l.28.28a6 6 0 0 1 1.76 4.24V17a3 3 0 0 1-3 3H9.74a3 3 0 0 1-3-3v-4.12a6 6 0 0 1 1.76-4.24l.28-.28A3.99 3.99 0 0 1 8 6a4 4 0 0 1 4-4Z"></path>
            </svg>
          </div>
          <div className="brand-info">
            <h1 className="brand-title">BreedID Livestock Vision</h1>
            <span className="brand-subtitle">AI Breed Classification & OOD Detection</span>
          </div>
        </div>

        <div className="header-nav-right">
          <div className="status-badge" title={connected ? "Backend Connected" : "Connecting..."}>
            <span className={`status-dot ${connected ? "status-online" : "status-offline"}`} />
            <span className="status-text">{connected ? "Connected" : "Reconnecting"}</span>
          </div>

          <div className="model-chip" title={`Active Model: ${activeModelDisplay}`}>
            <span className="chip-label">Model:</span>
            <span className="chip-value">{activeModelDisplay}</span>
          </div>

          <ThemeToggle />

          <Link to="/dev" className="btn btn-outline dev-portal-btn">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path>
              <circle cx="12" cy="12" r="3"></circle>
            </svg>
            Developer Settings
          </Link>
        </div>
      </header>

      {/* Main Content Body */}
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
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                Upload Photo
              </button>
              <button
                type="button"
                className={`tab-btn ${inputMode === "camera" ? "tab-active" : ""}`}
                onClick={() => setInputMode("camera")}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                  <circle cx="12" cy="13" r="4"></circle>
                </svg>
                Live Camera
              </button>
            </div>

            <div className="tab-pane-content">
              {inputMode === "upload" && (
                <ImageUpload
                  onImageSelected={handleImageSelected}
                  currentPreview={previewUrl}
                  onClear={handleClear}
                />
              )}

              {inputMode === "camera" && (
                <div className="camera-tab-container">
                  <CameraCapture
                    onCapture={handleCameraCapture}
                    onClose={() => setInputMode("upload")}
                  />
                </div>
              )}

              {selectedFile && !predictionResult && (
                <div className="action-button-row">
                  <button
                    type="button"
                    className="btn btn-primary btn-large btn-full"
                    disabled={analyzing}
                    onClick={handleRunPrediction}
                  >
                    {analyzing ? (
                      <span className="btn-spinner-content">
                        <span className="spinner-border" />
                        Analyzing with Vision Model...
                      </span>
                    ) : (
                      <>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <circle cx="11" cy="11" r="8"></circle>
                          <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                        </svg>
                        Identify Breed
                      </>
                    )}
                  </button>
                </div>
              )}

              {analyzing && predictionEvent?.type === "progress" && (
                <div className="progress-status-box">
                  <div className="pulse-dot" />
                  <span>{predictionEvent.detail || "Processing..."}</span>
                </div>
              )}

              {error && (
                <div className="error-banner">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                  </svg>
                  <span>{error}</span>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Prediction Results Display */}
        {predictionResult && (
          <section className="results-panel-section">
            <ResultCard
              result={predictionResult}
              config={config}
              onReset={handleClear}
            />
          </section>
        )}
      </main>

      <footer className="site-footer">
        <p>Cattle & Buffalo Breed Classifier &bull; ML Vision System &bull; Soft-routed Architecture</p>
      </footer>
    </div>
  );
}
