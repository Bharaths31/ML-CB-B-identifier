import { useState } from "react";
import { exportOdt, downloadBlob } from "../api.js";
import { readSession, sessionToOdtResults } from "../utils/sessionStore.js";

export default function DevConfigPanel({ config, onUpdateConfig, onResetConfig }) {
  const [exporting, setExporting] = useState(false);
  const [resetting, setResetting] = useState(false);

  const display = config?.display || {};

  const handleDisplayChange = async (key, val) => {
    try {
      await onUpdateConfig({ [`display.${key}`]: val });
    } catch (err) {
      alert("Failed to update config: " + err.message);
    }
  };

  const handleReset = async () => {
    if (!window.confirm("Are you sure you want to reset all application settings to defaults?")) {
      return;
    }
    try {
      setResetting(true);
      await onResetConfig();
      alert("Configuration has been reset to defaults.");
    } catch (err) {
      alert("Reset failed: " + err.message);
    } finally {
      setResetting(false);
    }
  };

  const handleExportOdt = async () => {
    try {
      setExporting(true);
      const entries = readSession();
      const results = sessionToOdtResults(entries);
      const blob = await exportOdt(
        results,
        results.length > 0 ? "batch" : "summary"
      );
      downloadBlob(blob, "classification_report.odt");
    } catch (err) {
      alert("Export failed: " + err.message);
    } finally {
      setExporting(false);
    }
  };

  const dataset = config?.dataset || {};

  const handleDatasetChange = async (key, val) => {
    try {
      await onUpdateConfig({ [`dataset.${key}`]: val });
    } catch (err) {
      alert("Failed to update dataset config: " + err.message);
    }
  };

  const applyBreedPreset = async (minB, maxB, minConf) => {
    try {
      await onUpdateConfig({
        "display.min_breeds": minB,
        "display.max_breeds": maxB,
        "display.top_k": maxB,
        "display.min_breed_confidence_pct": minConf,
      });
    } catch (err) {
      alert("Failed to apply preset: " + err.message);
    }
  };

  return (
    <div className="dev-panel-content">
      <div className="panel-header-block">
        <h2 className="panel-title">Display & Application Preferences</h2>
        <p className="panel-description">
          Adjust visual thresholds, min/max candidate breed counts, confidence filtering, and dataset parameters.
          All changes are saved to <code>outputs/app_config.json</code> and synced across browser sessions in real time.
        </p>
      </div>

      <div className="dev-grid-2col">
        {/* Left Column: UI Presentation & Breed Display Controls */}
        <div className="dev-card">
          <div className="dev-card-header">
            <h3 className="card-heading">Candidate Breed Display Limits</h3>
            <span className="badge badge-neutral">Shared Config</span>
          </div>

          {/* Quick Presets */}
          <div className="form-group" style={{ marginBottom: 12 }}>
            <label className="form-label" style={{ marginBottom: 4 }}>Quick Range Presets</label>
            <div className="preset-chip-row">
              <button
                type="button"
                className={`preset-chip ${(display.min_breeds === 1 && display.max_breeds === 1) ? "active" : ""}`}
                onClick={() => applyBreedPreset(1, 1, 10)}
              >
                Top 1 Strict
              </button>
              <button
                type="button"
                className={`preset-chip ${(display.min_breeds === 1 && display.max_breeds === 3) ? "active" : ""}`}
                onClick={() => applyBreedPreset(1, 3, 5)}
              >
                Standard (1–3)
              </button>
              <button
                type="button"
                className={`preset-chip ${(display.min_breeds === 2 && display.max_breeds === 5) ? "active" : ""}`}
                onClick={() => applyBreedPreset(2, 5, 5)}
              >
                Extended (2–5)
              </button>
              <button
                type="button"
                className={`preset-chip ${(display.min_breeds === 1 && (display.max_breeds === 10 || display.top_k === 10)) ? "active" : ""}`}
                onClick={() => applyBreedPreset(1, 10, 0)}
              >
                All (Up to 10)
              </button>
            </div>
          </div>

          {/* Min Breeds */}
          <div className="slider-group">
            <div className="slider-header">
              <label className="slider-label" htmlFor="min-breeds-slider">Min Candidate Breeds</label>
              <span className="slider-number-badge">{display.min_breeds ?? 1}</span>
            </div>
            <input
              id="min-breeds-slider"
              type="range"
              min="1"
              max="5"
              step="1"
              className="range-input"
              value={display.min_breeds ?? 1}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                const currentMax = display.max_breeds ?? (display.top_k ?? 3);
                const updates = { "display.min_breeds": val };
                if (val > currentMax) {
                  updates["display.max_breeds"] = val;
                  updates["display.top_k"] = val;
                }
                onUpdateConfig(updates);
              }}
            />
            <p className="slider-hint">
              Minimum number of breed candidates guaranteed to display (even if alternative probabilities are low).
            </p>
          </div>

          {/* Max Breeds / Top-K */}
          <div className="slider-group">
            <div className="slider-header">
              <label className="slider-label" htmlFor="max-breeds-slider">Max Candidate Breeds (Top-K)</label>
              <span className="slider-number-badge">{display.max_breeds ?? (display.top_k ?? 3)}</span>
            </div>
            <input
              id="max-breeds-slider"
              type="range"
              min="1"
              max="10"
              step="1"
              className="range-input"
              value={display.max_breeds ?? (display.top_k ?? 3)}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                const currentMin = display.min_breeds ?? 1;
                const updates = {
                  "display.max_breeds": val,
                  "display.top_k": val,
                };
                if (val < currentMin) {
                  updates["display.min_breeds"] = val;
                }
                onUpdateConfig(updates);
              }}
            />
            <p className="slider-hint">
              Maximum upper bound on alternative candidate breed predictions shown to the user.
            </p>
          </div>

          {/* Min Breed Confidence % Cutoff */}
          <div className="slider-group">
            <div className="slider-header">
              <label className="slider-label" htmlFor="min-confidence-slider">Min Breed Confidence Cutoff (%)</label>
              <span className="slider-number-badge">{display.min_breed_confidence_pct ?? 5}%</span>
            </div>
            <input
              id="min-confidence-slider"
              type="range"
              min="0"
              max="30"
              step="1"
              className="range-input"
              value={display.min_breed_confidence_pct ?? 5}
              onChange={(e) => handleDisplayChange("min_breed_confidence_pct", parseFloat(e.target.value))}
            />
            <p className="slider-hint">
              Omit alternative breed predictions whose confidence falls below this threshold (unless needed to satisfy Min Breeds).
            </p>
          </div>

          {/* Presentation & Formatting Settings */}
          <div className="form-group" style={{ marginTop: 16 }}>
            <label className="form-label" htmlFor="confidence-display-mode">Confidence Display Mode</label>
            <select
              id="confidence-display-mode"
              className="form-select"
              value={display.confidence_mode || "percentage"}
              onChange={(e) => handleDisplayChange("confidence_mode", e.target.value)}
            >
              <option value="percentage">Direct Percentage (Raw Softmax × Species Prob)</option>
              <option value="scaled">Presenter Scaled (Rescaled above minimum floor)</option>
              <option value="clamped">Floor Minimum % (Clamp low scores)</option>
              <option value="badge_only">Badge Only (High / Moderate / Low categorical badge)</option>
              <option value="hide">Hide Confidence (show status only)</option>
            </select>
          </div>

          <div className="slider-group">
            <div className="slider-header">
              <label className="slider-label" htmlFor="floor-slider">Confidence Floor (%)</label>
              <span className="slider-number-badge">{display.confidence_floor_pct ?? 80}%</span>
            </div>
            <input
              id="floor-slider"
              type="range"
              min="50"
              max="95"
              step="1"
              className="range-input"
              value={display.confidence_floor_pct ?? 80}
              onChange={(e) => handleDisplayChange("confidence_floor_pct", parseFloat(e.target.value))}
            />
            <p className="slider-hint">
              Used when confidence mode is set to "Presenter Scaled".
            </p>
          </div>

          {/* User-content toggles moved to the Vet Portal */}
          <div className="dev-subcard" style={{ marginTop: 16 }}>
            <div className="dev-card-header" style={{ marginBottom: 8 }}>
              <h4 className="card-heading" style={{ fontSize: "0.92rem" }}>User Page Visibility</h4>
              <span className="badge badge-neutral">Vet Portal</span>
            </div>
            <p className="slider-hint">
              Species badge/confidence, escalation warnings, technical details and
              section-level hidden content are now controlled in the Vet Portal’s
              <strong> User Page</strong> tab.
            </p>
            <a
              className="btn btn-outline"
              style={{ marginTop: 10 }}
              href={import.meta.env.VITE_VET_PORTAL_URL || "http://localhost:5174"}
              target="_blank"
              rel="noreferrer"
            >
              Open Vet Portal ↗
            </a>
          </div>

          {/* Dataset Thresholds */}
          <div className="dev-subcard" style={{ marginTop: 16 }}>
            <div className="dev-card-header" style={{ marginBottom: 12 }}>
              <h4 className="card-heading" style={{ fontSize: "0.92rem" }}>Dataset & Training Breed Thresholds</h4>
              <span className="badge badge-neutral">data/splits</span>
            </div>

            <div className="slider-group">
              <div className="slider-header">
                <label className="slider-label" htmlFor="min-imgs-slider">Min Images / Breed Cutoff</label>
                <span className="slider-number-badge">{dataset.min_images_per_breed ?? 15} imgs</span>
              </div>
              <input
                id="min-imgs-slider"
                type="range"
                min="0"
                max="50"
                step="5"
                className="range-input"
                value={dataset.min_images_per_breed ?? 15}
                onChange={(e) => handleDatasetChange("min_images_per_breed", parseInt(e.target.value, 10))}
              />
              <p className="slider-hint">
                Breeds with fewer total images than this threshold are dropped during dataset splitting to avoid severely under-supported classes.
              </p>
            </div>
          </div>
        </div>

        {/* Right Column: Database Snapshot & Actions */}
        <div className="dev-card">
          <div className="dev-card-header">
            <h3 className="card-heading">Shared Database State</h3>
            <span className="badge badge-outline">outputs/app_config.json</span>
          </div>

          <div className="json-preview-container">
            <pre className="json-preview-code">
              {JSON.stringify(config, null, 2)}
            </pre>
          </div>

          <div className="admin-actions-block">
            <h4 className="card-subheading">Maintenance & Reports</h4>
            <div className="actions-button-group">
              <button
                type="button"
                className="btn btn-secondary"
                disabled={exporting}
                onClick={handleExportOdt}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                {exporting ? "Generating ODT..." : "Export ODT Report"}
              </button>

              <button
                type="button"
                className="btn btn-danger"
                disabled={resetting}
                onClick={handleReset}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
                  <path d="M21 3v5h-5"></path>
                  <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
                  <path d="M3 21v-5h5"></path>
                </svg>
                Reset All Settings to Defaults
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
