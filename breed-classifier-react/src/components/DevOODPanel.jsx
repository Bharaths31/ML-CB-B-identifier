import { useState, useEffect } from "react";
import {
  getOODProfiles,
  getActiveOOD,
  selectOODProfile,
  toggleOOD,
  updateOODThresholds,
  uploadOODProfile,
  getModels,
  getOODDataDirs,
  recalibrateOOD,
} from "../api.js";

export default function DevOODPanel({ config, onConfigChanged }) {
  const [profiles, setProfiles] = useState([]);
  const [activeInfo, setActiveInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  // Editable threshold state
  const [energyThresh, setEnergyThresh] = useState(config?.ood?.energy_threshold ?? -5.0);
  const [mspThresh, setMspThresh] = useState(config?.ood?.msp_threshold ?? 0.5);
  const [temp, setTemp] = useState(config?.ood?.temperature ?? 1.0);
  const [savingThresholds, setSavingThresholds] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null);

  // Manual folder-based recalibration
  const [models, setModels] = useState([]);
  const [calModel, setCalModel] = useState(config?.active_model || "");
  const [dataDirs, setDataDirs] = useState([]);
  const [dataDir, setDataDir] = useState("");
  const [percentile, setPercentile] = useState(95);
  const [maxImages, setMaxImages] = useState(500);
  const [recalibrating, setRecalibrating] = useState(false);
  const [recalResult, setRecalResult] = useState(null);
  const [recalError, setRecalError] = useState(null);

  const isEnabled = config?.ood?.enabled ?? true;
  const selectedProfile = config?.ood?.selected_profile ?? "auto";

  const refreshData = async () => {
    try {
      setLoading(true);
      const [pRes, aRes] = await Promise.allSettled([getOODProfiles(), getActiveOOD()]);
      if (pRes.status === "fulfilled") {
        setProfiles(pRes.value.profiles || []);
      }
      if (aRes.status === "fulfilled") {
        setActiveInfo(aRes.value);
        if (aRes.value?.config) {
          setEnergyThresh(aRes.value.config.energy_threshold ?? -5.0);
          setMspThresh(aRes.value.config.msp_threshold ?? 0.5);
          setTemp(aRes.value.config.temperature ?? 1.0);
        }
      }
    } catch (err) {
      console.error("Failed to load OOD data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshData();
  }, [config]);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([getModels(), getOODDataDirs()]).then(([mRes, dRes]) => {
      if (cancelled) return;
      if (mRes.status === "fulfilled") {
        const list = mRes.value.models || [];
        setModels(list);
        setCalModel((current) => current || config?.active_model || list[0]?.name || "");
      }
      if (dRes.status === "fulfilled") {
        const dirs = dRes.value.dirs || [];
        setDataDirs(dirs);
        setDataDir((current) => current || dirs[0]?.path || "");
      }
    });
    return () => {
      cancelled = true;
    };
  }, [config?.active_model]);

  useEffect(() => {
    if (config?.active_model) setCalModel((current) => current || config.active_model);
  }, [config?.active_model]);

  const handleRecalibrate = async () => {
    if (!calModel) {
      setRecalError("Select a model first.");
      return;
    }
    try {
      setRecalibrating(true);
      setRecalError(null);
      setRecalResult(null);
      const result = await recalibrateOOD({
        model: calModel,
        data_dir: dataDir.trim() || undefined,
        percentile: Number(percentile) || 95,
        max_images: Number(maxImages) || undefined,
      });
      setRecalResult(result);
      onConfigChanged?.();
      await refreshData();
    } catch (err) {
      setRecalError(err.message || "Recalibration failed");
    } finally {
      setRecalibrating(false);
    }
  };

  const handleToggle = async (e) => {
    const nextVal = e.target.checked;
    try {
      await toggleOOD(nextVal);
      onConfigChanged?.();
    } catch (err) {
      alert("Failed to toggle OOD: " + err.message);
    }
  };

  const handleProfileSelect = async (e) => {
    const prof = e.target.value;
    try {
      await selectOODProfile(prof);
      onConfigChanged?.();
      await refreshData();
    } catch (err) {
      alert("Failed to select profile: " + err.message);
    }
  };

  const handleSaveThresholds = async () => {
    try {
      setSavingThresholds(true);
      await updateOODThresholds({
        energy_threshold: parseFloat(energyThresh),
        msp_threshold: parseFloat(mspThresh),
        temperature: parseFloat(temp),
      });
      onConfigChanged?.();
      alert("OOD thresholds updated successfully!");
    } catch (err) {
      alert("Failed to update thresholds: " + err.message);
    } finally {
      setSavingThresholds(false);
    }
  };

  const handleUploadProfile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setUploadMsg("Uploading profile...");
      const res = await uploadOODProfile(file);
      if (res.error) throw new Error(res.error);
      setUploadMsg(`Uploaded ${res.filename} successfully!`);
      await refreshData();
    } catch (err) {
      setUploadMsg(`Upload failed: ${err.message}`);
    }
  };

  return (
    <div className="dev-panel-content">
      <div className="panel-header-block">
        <h2 className="panel-title">Out-of-Distribution (OOD) Calibration</h2>
        <p className="panel-description">
          Detect and reject outlier images (e.g. non-livestock, unknown breeds, corrupted images)
          using post-hoc energy scoring without model retraining.
        </p>
      </div>

      <div className="dev-grid-2col">
        {/* Left Column: Master Toggle & Profile Selection */}
        <div className="dev-card">
          <div className="dev-card-header">
            <h3 className="card-heading">OOD Detection Control</h3>
            <span className={`badge ${isEnabled ? "badge-success" : "badge-neutral"}`}>
              {isEnabled ? "ACTIVE" : "BYPASSED"}
            </span>
          </div>

          <div className="control-setting-row">
            <div className="setting-info">
              <span className="setting-label">Enable OOD Filtering</span>
              <span className="setting-subtext">
                When disabled, all images pass directly to soft breed routing.
              </span>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={isEnabled}
                onChange={handleToggle}
              />
              <span className="toggle-slider" />
            </label>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="ood-profile-select">Calibration Profile</label>
            <select
              id="ood-profile-select"
              className="form-select"
              value={selectedProfile}
              onChange={handleProfileSelect}
              disabled={!isEnabled}
            >
              <option value="auto">Auto-resolve per active model (outputs/ood/)</option>
              {profiles.map((p) => (
                <option key={p.filename} value={p.filename}>
                  {p.filename} ({p.model_name})
                </option>
              ))}
            </select>
            <span className="form-hint">
              Auto resolves calibrated energy thresholds matched to the active checkpoint.
            </span>
          </div>

          {/* Active Profile Info Box */}
          <div className="active-profile-summary">
            <h4 className="summary-title">Runtime Detector Status</h4>
            <div className="summary-row">
              <span className="summary-k">Status:</span>
              <span className="summary-v">{activeInfo?.detector?.initialized ? "Loaded in Memory" : "Not Initialized"}</span>
            </div>
            <div className="summary-row">
              <span className="summary-k">Loaded Profile:</span>
              <span className="summary-v">{activeInfo?.detector?.profile_name || "None"}</span>
            </div>
            <div className="summary-row">
              <span className="summary-k">Source File:</span>
              <span className="summary-v text-truncate">{activeInfo?.detector?.source_file || "Config Fallback"}</span>
            </div>
          </div>

          {/* Upload New Profile */}
          <div className="model-upload-box">
            <label className="upload-btn-label">
              <input
                type="file"
                accept=".json"
                style={{ display: "none" }}
                onChange={handleUploadProfile}
              />
              <span className="btn btn-secondary btn-full">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                Upload Calibration Profile (.json)
              </span>
            </label>
            {uploadMsg && <p className="upload-status-text">{uploadMsg}</p>}
          </div>
        </div>

        {/* Right Column: Threshold Fine-Tuning */}
        <div className="dev-card">
          <div className="dev-card-header">
            <h3 className="card-heading">Threshold Hyperparameters</h3>
            <span className="badge badge-outline">Live Tuning</span>
          </div>

          <div className="threshold-controls-form">
            <div className="slider-group">
              <div className="slider-header">
                <label className="slider-label">Energy Threshold</label>
                <span className="slider-number-badge">{Number(energyThresh).toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-15"
                max="0"
                step="0.1"
                className="range-input"
                value={energyThresh}
                onChange={(e) => setEnergyThresh(e.target.value)}
                disabled={!isEnabled}
              />
              <p className="slider-hint">
                Samples with free energy above this cut-off are marked Out-of-Distribution.
              </p>
            </div>

            <div className="slider-group">
              <div className="slider-header">
                <label className="slider-label">MSP Threshold (Softmax Floor)</label>
                <span className="slider-number-badge">{Number(mspThresh).toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                className="range-input"
                value={mspThresh}
                onChange={(e) => setMspThresh(e.target.value)}
                disabled={!isEnabled}
              />
              <p className="slider-hint">
                Minimum maximum softmax probability required to pass validation.
              </p>
            </div>

            <div className="slider-group">
              <div className="slider-header">
                <label className="slider-label">Energy Temperature (T)</label>
                <span className="slider-number-badge">{Number(temp).toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="3.0"
                step="0.1"
                className="range-input"
                value={temp}
                onChange={(e) => setTemp(e.target.value)}
                disabled={!isEnabled}
              />
              <p className="slider-hint">
                Temperature scaling factor applied to raw logits prior to computing LogSumExp.
              </p>
            </div>

            <div className="threshold-actions">
              <button
                type="button"
                className="btn btn-primary btn-full"
                onClick={handleSaveThresholds}
                disabled={!isEnabled || savingThresholds}
              >
                {savingThresholds ? "Applying Changes..." : "Apply & Persist Thresholds"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Manual folder-based recalibration */}
      <div className="dev-card ood-recal-card">
        <div className="dev-card-header">
          <h3 className="card-heading">Manual OOD Recalibration</h3>
          <span className="badge badge-outline">Breed-named folder</span>
        </div>
        <p className="panel-description">
          Pick the model and a folder of breed-named images. Flat files
          (<code>Deoni_01.jpg</code>) and nested folders (<code>cattle/deoni/…</code>)
          are both supported. Only names matching the model class maps are used;
          unmatched files are reported instead of silently polluting thresholds.
        </p>

        <div className="ood-recal-grid">
          <label className="form-group">
            <span className="form-label">Model</span>
            <select
              className="form-select"
              value={calModel}
              onChange={(e) => setCalModel(e.target.value)}
            >
              {models.length === 0 && <option value="">No models found</option>}
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} ({m.backbone})
                </option>
              ))}
            </select>
          </label>

          <label className="form-group">
            <span className="form-label">Breed-named image folder</span>
            <input
              className="form-input"
              list="ood-data-dir-options"
              value={dataDir}
              onChange={(e) => setDataDir(e.target.value)}
              placeholder="data/raw or Testing data"
            />
            <datalist id="ood-data-dir-options">
              {dataDirs.map((d) => (
                <option key={d.abs_path || d.path} value={d.path}>
                  {d.label}
                </option>
              ))}
            </datalist>
          </label>

          <label className="form-group">
            <span className="form-label">Percentile</span>
            <input
              className="form-input"
              type="number"
              min="50"
              max="99.9"
              step="0.5"
              value={percentile}
              onChange={(e) => setPercentile(e.target.value)}
            />
          </label>

          <label className="form-group">
            <span className="form-label">Max images</span>
            <input
              className="form-input"
              type="number"
              min="1"
              value={maxImages}
              onChange={(e) => setMaxImages(e.target.value)}
            />
          </label>
        </div>

        <button
          type="button"
          className="btn btn-primary btn-full"
          onClick={handleRecalibrate}
          disabled={recalibrating || !calModel}
        >
          {recalibrating ? "Calibrating from folder…" : "Recalibrate OOD from Folder"}
        </button>

        {recalError && <p className="ood-recal-error">{recalError}</p>}

        {recalResult && (
          <div className="ood-recal-result">
            <div className="ood-recal-stats">
              <div>
                <span>Samples</span>
                <strong>
                  {recalResult.calibrated_on_samples ?? recalResult.n_samples ?? "—"}
                </strong>
              </div>
              <div>
                <span>Matched</span>
                <strong>{recalResult.matched_images ?? "—"}</strong>
              </div>
              <div>
                <span>Unmatched</span>
                <strong>{recalResult.unmatched_images ?? "—"}</strong>
              </div>
              <div>
                <span>Energy</span>
                <strong>
                  {typeof recalResult.energy_threshold === "number"
                    ? recalResult.energy_threshold.toFixed(4)
                    : "—"}
                </strong>
              </div>
              <div>
                <span>MSP</span>
                <strong>
                  {typeof recalResult.msp_threshold === "number"
                    ? recalResult.msp_threshold.toFixed(4)
                    : "—"}
                </strong>
              </div>
              <div>
                <span>Percentile</span>
                <strong>{recalResult.percentile ?? "—"}</strong>
              </div>
            </div>

            {recalResult.breed_counts &&
              Object.keys(recalResult.breed_counts).length > 0 && (
                <div className="ood-breed-chips">
                  {Object.entries(recalResult.breed_counts)
                    .slice(0, 16)
                    .map(([name, count]) => (
                      <span className="ood-chip" key={name}>
                        {name} <strong>{count}</strong>
                      </span>
                    ))}
                </div>
              )}

            {recalResult.unknown_breeds?.length > 0 && (
              <p className="ood-recal-warning">
                Unmatched names:{" "}
                {recalResult.unknown_breeds
                  .slice(0, 8)
                  .map((x) => `${x.name} (${x.count})`)
                  .join(", ")}
              </p>
            )}

            {recalResult.path && (
              <p className="ood-recal-path">
                Saved profile: <code>{recalResult.path}</code>
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
