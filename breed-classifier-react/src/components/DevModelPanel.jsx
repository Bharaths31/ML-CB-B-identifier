import { useState, useEffect } from "react";
import { getModels, selectModel, uploadModel, getModelSpec, embedModel, testModel } from "../api.js";

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

  // Developer model tools
  const [embedOod, setEmbedOod] = useState(true);
  const [embedCopy, setEmbedCopy] = useState(true);
  const [embedding, setEmbedding] = useState(false);
  const [embedResult, setEmbedResult] = useState(null);
  const [embedError, setEmbedError] = useState(null);
  const [testFile, setTestFile] = useState(null);
  const [testRepeats, setTestRepeats] = useState(1);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [testError, setTestError] = useState(null);

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

  const handleEmbed = async () => {
    if (!selectedSpecModel) return;
    try {
      setEmbedding(true);
      setEmbedError(null);
      setEmbedResult(null);
      const res = await embedModel({
        model: selectedSpecModel,
        embedOod,
        copy: embedCopy,
      });
      setEmbedResult(res);
      await loadModelList();
      if (res?.model) setSelectedSpecModel(res.model);
      else setSelectedSpecModel(selectedSpecModel);
    } catch (err) {
      setEmbedError(err.message || "Embed failed");
    } finally {
      setEmbedding(false);
    }
  };

  const handleTest = async () => {
    if (!selectedSpecModel || !testFile) return;
    try {
      setTesting(true);
      setTestError(null);
      setTestResult(null);
      const res = await testModel({
        model: selectedSpecModel,
        file: testFile,
        repeats: testRepeats,
      });
      setTestResult(res);
    } catch (err) {
      setTestError(err.message || "Model test failed");
    } finally {
      setTesting(false);
    }
  };

  const metadataOod = spec?.embedded_metadata?.ood || spec?.ood;
  const oodLabel = metadataOod?.energy_threshold != null
    ? `E ${Number(metadataOod.energy_threshold).toFixed(3)} · MSP ${Number(metadataOod.msp_threshold).toFixed(3)}`
    : "Defaults / legacy JSON";

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

          {/* Model-specific test */}
          <div className="dev-subcard" style={{ marginTop: 16 }}>
            <div className="dev-card-header" style={{ marginBottom: 12 }}>
              <h4 className="card-heading" style={{ fontSize: "0.92rem" }}>Test This Model</h4>
              <span className="badge badge-neutral">{selectedSpecModel || "none"}</span>
            </div>
            <p className="slider-hint" style={{ marginBottom: 12 }}>
              Runs only the selected model (no advisory/CLIP) and reports raw
              softmax routing, OOD and latency.
            </p>
            <label className="upload-btn-label">
              <input
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => setTestFile(e.target.files?.[0] || null)}
                disabled={testing}
              />
              <span className="btn btn-outline btn-full">
                {testFile ? `Image: ${testFile.name}` : "Choose a test image"}
              </span>
            </label>

            <div className="form-group" style={{ marginTop: 10 }}>
              <label className="form-label" htmlFor="model-test-repeats">Repeats (latency)</label>
              <select
                id="model-test-repeats"
                className="form-input"
                value={testRepeats}
                onChange={(e) => setTestRepeats(parseInt(e.target.value, 10))}
              >
                {[1, 3, 5, 10].map((n) => <option key={n} value={n}>{n}×</option>)}
              </select>
            </div>

            <button
              type="button"
              className="btn btn-primary btn-full"
              style={{ marginTop: 10 }}
              onClick={handleTest}
              disabled={!selectedSpecModel || !testFile || testing}
            >
              {testing ? "Running model test…" : "Run model test"}
            </button>

            {testError && <p className="camera-error-inline">{testError}</p>}
            {testResult && (
              <div className="spec-details-table" style={{ marginTop: 12 }}>
                <div className="spec-row">
                  <span className="spec-k">Prediction</span>
                  <span className="spec-v">
                    {testResult.species} · {testResult.breed} ({testResult.confidence}%)
                  </span>
                </div>
                <div className="spec-row">
                  <span className="spec-k">OOD</span>
                  <span className="spec-v">
                    {testResult.is_ood ? "ABSTAIN" : "in-distribution"}
                    {` · ${Number(testResult.ood_score).toFixed(3)}`}
                  </span>
                </div>
                <div className="spec-row">
                  <span className="spec-k">Latency</span>
                  <span className="spec-v">
                    {testResult.latency_ms?.mean} ms mean · {testResult.latency_ms?.last} ms last
                  </span>
                </div>
                {testResult.top5?.map((item, idx) => (
                  <div className="spec-row" key={`${item.breed_slug}-${idx}`}>
                    <span className="spec-k">#{idx + 1} {item.breed}</span>
                    <span className="spec-v">{item.confidence}%</span>
                  </div>
                ))}
              </div>
            )}
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

              <div className="spec-tensor-section">
                <h4 className="spec-subheading">Self-Describing Metadata</h4>
                <div className="spec-row">
                  <span className="spec-k">Status</span>
                  <span className="spec-v">
                    {spec.self_describing ? "Embedded" : "Not embedded (runtime introspection)"}
                  </span>
                </div>
                <div className="spec-row">
                  <span className="spec-k">Class map source</span>
                  <span className="spec-v text-truncate">{spec.class_map_source || "—"}</span>
                </div>
                <div className="spec-row">
                  <span className="spec-k">Normalization</span>
                  <span className="spec-v">
                    {spec.normalization_baked ? "Baked in graph ([0,1] input)" : "Caller-side ImageNet"}
                  </span>
                </div>
                <div className="spec-row">
                  <span className="spec-k">OOD thresholds</span>
                  <span className="spec-v">{oodLabel}</span>
                </div>
                {spec.class_counts && (
                  <div className="spec-row">
                    <span className="spec-k">Class counts</span>
                    <span className="spec-v">
                      {spec.class_counts.cattle} cattle · {spec.class_counts.buffalo} buffalo
                    </span>
                  </div>
                )}
                {spec.embedded_metadata && Object.keys(spec.embedded_metadata).length > 0 && (
                  <pre className="json-preview-code" style={{ maxHeight: 170, marginTop: 8 }}>
                    {JSON.stringify(spec.embedded_metadata, null, 2)}
                  </pre>
                )}
              </div>

              <div className="spec-tensor-section">
                <h4 className="spec-subheading">Embed Metadata</h4>
                <p className="slider-hint" style={{ marginBottom: 10 }}>
                  Writes class maps, resolution, normalization and (optionally) OOD
                  thresholds into the model file. Weights are never modified.
                </p>
                <div className="control-setting-row">
                  <div className="setting-info">
                    <span className="setting-label">Embed OOD thresholds</span>
                    <span className="setting-subtext">Bake the current calibrated thresholds into the file.</span>
                  </div>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={embedOod}
                      onChange={(e) => setEmbedOod(e.target.checked)}
                    />
                    <span className="toggle-slider" />
                  </label>
                </div>
                <div className="control-setting-row">
                  <div className="setting-info">
                    <span className="setting-label">Write a copy (recommended)</span>
                    <span className="setting-subtext">Saves under outputs/export/embedded/ and leaves the original intact.</span>
                  </div>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={embedCopy}
                      onChange={(e) => setEmbedCopy(e.target.checked)}
                    />
                    <span className="toggle-slider" />
                  </label>
                </div>
                <button
                  type="button"
                  className="btn btn-primary btn-full"
                  style={{ marginTop: 10 }}
                  onClick={handleEmbed}
                  disabled={!selectedSpecModel || embedding || specLoading}
                >
                  {embedding ? "Embedding metadata…" : "Embed metadata into model"}
                </button>
                {embedError && <p className="camera-error-inline">{embedError}</p>}
                {embedResult && (
                  <p className="upload-status-text">
                    Embedded {embedResult.ood_embedded ? "with OOD" : "without OOD"} →{" "}
                    {embedResult.path}
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
