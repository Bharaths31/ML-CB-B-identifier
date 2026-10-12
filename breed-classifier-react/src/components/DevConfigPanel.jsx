import { useState } from "react";
import { exportOdt, downloadBlob } from "../api.js";

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
      const blob = await exportOdt([], "summary");
      downloadBlob(blob, "classification_report.odt");
    } catch (err) {
      alert("Export failed: " + err.message);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="dev-panel-content">
      <div className="panel-header-block">
        <h2 className="panel-title">Display & Application Preferences</h2>
        <p className="panel-description">
          Adjust visual thresholds, confidence scaling, and presentation toggles.
          All changes are saved to <code>outputs/app_config.json</code> and synced across browser sessions.
        </p>
      </div>

      <div className="dev-grid-2col">
        {/* Left Column: UI Presentation Controls */}
        <div className="dev-card">
          <div className="dev-card-header">
            <h3 className="card-heading">User Interface Display Rules</h3>
            <span className="badge badge-neutral">Shared Config</span>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="confidence-display-mode">Confidence Display Mode</label>
            <select
              id="confidence-display-mode"
              className="form-select"
              value={display.confidence_mode || "percentage"}
              onChange={(e) => handleDisplayChange("confidence_mode", e.target.value)}
            >
              <option value="percentage">Direct Percentage (Raw Softmax × Species Prob)</option>
              <option value="scaled">Presenter Scaled (Rescaled above minimum floor)</option>
              <option value="badge_only">Badge Only (High / Moderate / Low categorical badge)</option>
            </select>
          </div>

          <div className="slider-group">
            <div className="slider-header">
              <label className="slider-label">Confidence Floor (%)</label>
              <span className="slider-number-badge">{display.confidence_floor_pct ?? 80}%</span>
            </div>
            <input
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

          <div className="slider-group">
            <div className="slider-header">
              <label className="slider-label">Top-K Ranked Breeds</label>
              <span className="slider-number-badge">{display.top_k ?? 3}</span>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              step="1"
              className="range-input"
              value={display.top_k ?? 3}
              onChange={(e) => handleDisplayChange("top_k", parseInt(e.target.value, 10))}
            />
            <p className="slider-hint">
              Maximum number of candidate breed predictions displayed to the user.
            </p>
          </div>

          <div className="control-setting-row">
            <div className="setting-info">
              <span className="setting-label">Show Species Confidence Tag</span>
              <span className="setting-subtext">Display Cattle vs Buffalo probability badge.</span>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={display.show_species_confidence ?? true}
                onChange={(e) => handleDisplayChange("show_species_confidence", e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          <div className="control-setting-row">
            <div className="setting-info">
              <span className="setting-label">Always Show Technical Details</span>
              <span className="setting-subtext">Expand latency and tensor info automatically.</span>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={display.show_technical_details ?? false}
                onChange={(e) => handleDisplayChange("show_technical_details", e.target.checked)}
              />
              <span className="toggle-slider" />
            </label>
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
