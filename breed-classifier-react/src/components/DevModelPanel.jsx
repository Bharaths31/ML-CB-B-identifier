import { useState, useEffect } from "react";
import { getModels, selectModel, uploadModel, getModelSpec } from "../api.js";

export default function DevModelPanel({ activeModel, onModelChanged }) {
  const [models, setModels] = useState([]);
  const [device, setDevice] = useState("cpu");
  const [loading, setLoading] = useState(true);
  const [selectedSpecModel, setSelectedSpecModel] = useState("");
  const [spec, setSpec] = useState(null);
  const [specLoading, setSpecLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState(null);
  const [activeName, setActiveName] = useState(activeModel || "");
  const [selecting, setSelecting] = useState("");
  const [selectMessage, setSelectMessage] = useState(null);
  const [loadError, setLoadError] = useState(null);

  const loadModelList = async () => {
    try {
      setLoading(true);
      setLoadError(null);
      const data = await getModels();
      setModels(data.models || []);
      setDevice(data.device || "cpu");
      const current = data.active_model || activeModel || data.models?.[0]?.name || "";
      if (current) setActiveName(current);
      if (current && !selectedSpecModel) {
        setSelectedSpecModel(current);
      }
    } catch (err) {
      setLoadError(err.message || "Failed to load models");
      console.error("Failed to load models:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModelList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeModel]);

  useEffect(() => {
    if (activeModel) setActiveName(activeModel);
  }, [activeModel]);

  useEffect(() => {
    if (!selectedSpecModel) return;
    const fetchSpec = async () => {
      try {
        setSpecLoading(true);
        const data = await getModelSpec(selectedSpecModel);
        setSpec(data);
      } catch (err) {
        setSpec(null);
      } finally {
        setSpecLoading(false);
      }
    };
    fetchSpec();
  }, [selectedSpecModel]);

  const handleSelectActive = async (modelName) => {
    if (selecting || modelName === activeName) return;
    setSelecting(modelName);
    setSelectMessage(null);
    try {
      const res = await selectModel(modelName);
      const nextActive = res?.active_model || modelName;
      setActiveName(nextActive);
      setSelectedSpecModel(nextActive);
      setSelectMessage(`Active model switched to ${nextActive}.`);
      if (typeof onModelChanged === "function") {
        await onModelChanged(nextActive);
      }
      await loadModelList();
      setActiveName(nextActive);
      setSelectedSpecModel(nextActive);
    } catch (err) {
      setSelectMessage(`Failed to switch model: ${err.message}`);
    } finally {
      setSelecting("");
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setUploading(true);
      setUploadMessage("Uploading checkpoint...");
      await uploadModel(file, file.name);
      setUploadMessage("Model uploaded successfully!");
      await loadModelList();
    } catch (err) {
      setUploadMessage("Upload failed: " + err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="dev-panel-content">
      <div className="panel-header-block">
        <h2 className="panel-title">Model Management & Checkpoints</h2>
        <p className="panel-description">
          Inspect, switch, and deploy backbone weights. Active model selection is immediately
          persisted to the shared database and broadcast to all connected clients.
        </p>
      </div>

      <div className="dev-grid-2col">
        {/* Left Column: Discovered Models */}
        <div className="dev-card">
          <div className="dev-card-header">
            <h3 className="card-heading">Available Models ({models.length})</h3>
            <span className="badge badge-neutral">Device: {device.toUpperCase()}</span>
          </div>

          <div className="model-list-container">
            {loading ? (
              <p className="text-muted">Scanning model directory...</p>
            ) : models.length === 0 ? (
              <p className="text-muted">No models found in outputs/ directory.</p>
            ) : (
              models.map((m) => {
                const isActive = (activeName || activeModel || "") === m.name;
                return (
                  <div key={m.name} className={`model-item-card ${isActive ? "model-active-card" : ""}`}>
                    <div className="model-item-info">
                      <div className="model-name-row">
                        <span className="model-name-title">{m.name}</span>
                        {isActive && <span className="badge badge-success">ACTIVE</span>}
                      </div>
                      <div className="model-meta-row">
                        <span>Backbone: {m.backbone}</span>
                        <span>Size: {m.size_mb} MB</span>
                      </div>
                    </div>

                    <div className="model-item-actions">
                      <button
                        type="button"
                        className="btn btn-sm btn-outline"
                        onClick={() => setSelectedSpecModel(m.name)}
                      >
                        Inspect
                      </button>
                      {!isActive && (
                        <button
                          type="button"
                          className="btn btn-sm btn-primary"
                          onClick={() => handleSelectActive(m.name)}
                          disabled={Boolean(selecting)}
                        >
                          {selecting === m.name ? "Activating…" : "Activate"}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {loadError && <p className="camera-error-inline">{loadError}</p>}
          {selectMessage && <p className="upload-status-text">{selectMessage}</p>}

          {/* Model Upload Section */}
          <div className="model-upload-box">
            <label className="upload-btn-label">
              <input
                type="file"
                accept=".pt,.onnx"
                style={{ display: "none" }}
                onChange={handleFileUpload}
                disabled={uploading}
              />
              <span className="btn btn-secondary btn-full">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                {uploading ? "Uploading..." : "Upload New Checkpoint (.pt / .onnx)"}
              </span>
            </label>
            {uploadMessage && <p className="upload-status-text">{uploadMessage}</p>}
          </div>
        </div>

        {/* Right Column: Model Spec Inspector */}
        <div className="dev-card">
          <div className="dev-card-header">
            <h3 className="card-heading">Architecture & Weights Inspector</h3>
            {spec && <span className="badge badge-outline">{spec.backbone}</span>}
          </div>

          {specLoading ? (
            <p className="text-muted">Loading model architecture parameters...</p>
          ) : !spec ? (
            <p className="text-muted">Select a model on the left to inspect architecture.</p>
          ) : (
            <div className="spec-details-table">
              <div className="spec-row">
                <span className="spec-k">Checkpoint Name</span>
                <span className="spec-v text-truncate">{spec.name}</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">Format / Type</span>
                <span className="spec-v">{spec.type?.toUpperCase()}</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">Total Parameters</span>
                <span className="spec-v">{spec.total_parameters_human || "N/A"}</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">Input Resolution</span>
                <span className="spec-v">{spec.image_size} &times; {spec.image_size} px</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">Weights File Size</span>
                <span className="spec-v">{spec.model_size_mb ? `${spec.model_size_mb} MB` : "N/A"}</span>
              </div>
              <div className="spec-row">
                <span className="spec-k">File Path</span>
                <span className="spec-v spec-path-value text-truncate">{spec.path}</span>
              </div>

              {spec.inputs && (
                <div className="spec-tensor-section">
                  <h4 className="spec-subheading">Input Nodes</h4>
                  {spec.inputs.map((inp, idx) => (
                    <div key={idx} className="tensor-node-row">
                      <code>{inp.name}</code> &bull; <span>{inp.shape}</span>
                    </div>
                  ))}
                </div>
              )}

              {spec.outputs && (
                <div className="spec-tensor-section">
                  <h4 className="spec-subheading">Output Heads</h4>
                  {spec.outputs.map((out, idx) => (
                    <div key={idx} className="tensor-node-row">
                      <code>{out.name}</code> &bull; <span>{out.shape}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
