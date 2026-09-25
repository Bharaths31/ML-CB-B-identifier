import { useEffect, useRef, useState } from "react";
import {
  getDevInfo, getModelSpec, getImageInfo, getPresenterConfig,
  savePresenterConfig, getLogs,
} from "../api.js";

function human(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function KV({ rows }) {
  return (
    <div className="dev-kv">
      {rows.map(([k, v]) => (
        <span key={k} style={{ display: "contents" }}>
          <span className="k">{k}</span>
          <span className="v">{String(v)}</span>
        </span>
      ))}
    </div>
  );
}

function JsonBlock({ obj }) {
  return <pre className="dev-json">{JSON.stringify(obj, null, 2)}</pre>;
}

function Collapsible({ title, children, defaultOpen }) {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <div>
      <div className="collapsible-header" onClick={() => setOpen(!open)}>
        {title} <span>{open ? "▲" : "▼"}</span>
      </div>
      {open && <div className="collapsible-body open">{children}</div>}
    </div>
  );
}

/* ---------------- Model Specification ---------------- */
function ModelSpec({ models }) {
  const [sel, setSel] = useState(models[0]?.name || "");
  const [spec, setSpec] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!sel && models.length) setSel(models[0].name);
  }, [models, sel]);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await getModelSpec(sel);
      if (s.error) setError(s.error);
      else setSpec(s);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>🔬 Model Specification</h2>
      <div className="model-selector">
        <label>Inspect Model</label>
        <select value={sel} onChange={(e) => setSel(e.target.value)}>
          {models.map((m) => (
            <option key={m.name} value={m.name}>{m.name}</option>
          ))}
        </select>
      </div>
      <button className="btn primary" onClick={load} disabled={loading || !sel}>
        {loading ? "⏳ Loading…" : "📋 Load Specification"}
      </button>
      <div className="dev-panel">
        {error && <div className="dev-placeholder" style={{ color: "var(--red)" }}>{error}</div>}
        {!spec && !error && <div className="dev-placeholder">Select a model and click <strong>Load Specification</strong></div>}
        {spec && (
          <>
            <div className="dev-section">
              <h3>📋 Overview</h3>
              <KV rows={[
                ["Name", spec.name], ["Backbone", spec.backbone], ["Type", spec.type],
                ["Device", spec.device],
                ["Input Size", `${spec.image_size || 260}×${spec.image_size || 260} RGB`],
                ["Path", spec.path],
              ]} />
            </div>
            {spec.total_parameters && (
              <div className="dev-section">
                <h3>📊 Parameters</h3>
                <KV rows={[
                  ["Total", `${spec.total_parameters_human} (${spec.total_parameters.toLocaleString()})`],
                  ["Trainable", spec.trainable_parameters.toLocaleString()],
                  ["Frozen", spec.frozen_parameters.toLocaleString()],
                  ["Est. Size", `${spec.model_size_mb} MB (float32)`],
                ]} />
              </div>
            )}
            {spec.architecture && (
              <div className="dev-section">
                <h3>🏗️ Architecture</h3>
                <KV rows={Object.entries(spec.architecture).map(([k, v]) => [k.replace(/_/g, " "), v])} />
              </div>
            )}
            {spec.components && (
              <div className="dev-section">
                <h3>🧩 Components</h3>
                <table className="dev-comp-table">
                  <thead><tr><th>Component</th><th>Params</th><th>Trainable</th></tr></thead>
                  <tbody>
                    {Object.entries(spec.components).map(([n, info]) => (
                      <tr key={n}><td>{n}</td><td>{info.params_human}</td><td>{human(info.trainable)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {spec.model_info_json && (
              <div className="dev-section">
                <h3>📝 model_info.json</h3>
                <JsonBlock obj={spec.model_info_json} />
              </div>
            )}
            {spec.inputs && (
              <div className="dev-section">
                <h3>ONNX I/O</h3>
                <JsonBlock obj={{ inputs: spec.inputs, outputs: spec.outputs }} />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/* ---------------- Class Maps ---------------- */
function ClassMaps() {
  const [info, setInfo] = useState(null);
  useEffect(() => {
    getDevInfo().then(setInfo).catch(() => setInfo({}));
  }, []);
  const cattle = info?.cattle_classes ? Object.keys(info.cattle_classes).length : 0;
  const buffalo = info?.buffalo_classes ? Object.keys(info.buffalo_classes).length : 0;
  return (
    <div className="card">
      <h2>📚 Class Maps (Breed Data)</h2>
      <div className="dev-panel">
        {!info && <div className="dev-placeholder">Loading class maps…</div>}
        {info && (
          <>
            <Collapsible title={`🐄 Cattle Breeds (${cattle})`}>
              <JsonBlock obj={info.cattle_classes || {}} />
            </Collapsible>
            <Collapsible title={`🐃 Buffalo Breeds (${buffalo})`}>
              <JsonBlock obj={info.buffalo_classes || {}} />
            </Collapsible>
            {cattle === 0 && buffalo === 0 && (
              <div className="dev-placeholder">No class maps found</div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/* ---------------- Image Metadata ---------------- */
function ImageMeta() {
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const analyze = async (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setBusy(true);
    setError(null);
    try {
      setMeta(await getImageInfo(f, f.name));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const exifKeys = meta?.exif ? Object.keys(meta.exif) : [];

  return (
    <div className="card">
      <h2>🖼️ Image Metadata Analyzer</h2>
      <div
        className={"dropzone" + (dragOver ? " drag-over" : "")}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); analyze(e.dataTransfer.files[0]); }}
      >
        <span className="icon">📐</span>
        <p>{busy ? "Analyzing…" : "Drop an image to inspect metadata"}<br />EXIF, dimensions, color space, and more</p>
        <div className="formats">Supports: PNG, JPG, JPEG, BMP, WebP</div>
        <input ref={inputRef} type="file" accept=".png,.jpg,.jpeg,.bmp,.webp" hidden
          onChange={(e) => analyze(e.target.files[0])} />
      </div>
      <div className="dev-panel">
        {error && <div className="dev-placeholder" style={{ color: "var(--red)" }}>{error}</div>}
        {!meta && !error && <div className="dev-placeholder">No image analyzed yet</div>}
        {meta && (
          <div className="meta-grid">
            <div className="meta-card">
              <h4>📐 Dimensions</h4>
              <KV rows={[
                ["Width", `${meta.width}px`], ["Height", `${meta.height}px`],
                ["Aspect Ratio", meta.aspect_ratio_simplified || meta.aspect_ratio],
                ["Megapixels", `${meta.megapixels} MP`],
                ["Orientation", meta.orientation],
              ]} />
            </div>
            <div className="meta-card">
              <h4>🎨 Color Info</h4>
              <KV rows={[
                ["Color Space", meta.color_space], ["Mode", meta.mode],
                ["Channels", `${meta.channels} (${(meta.bands || []).join(", ")})`],
                ["Bit Depth", `${meta.bit_depth}-bit`],
                ["Unique Colors", meta.unique_colors],
              ]} />
            </div>
            <div className="meta-card">
              <h4>💾 File Info</h4>
              <KV rows={[
                ["Format", meta.format],
                ["Size", `${meta.file_size_kb} KB (${meta.file_size_bytes} bytes)`],
                ["DPI", meta.dpi_x ? `${meta.dpi_x} × ${meta.dpi_y}` : "N/A"],
              ]} />
            </div>
            {exifKeys.length > 0 && (
              <div className="meta-card span-2">
                <h4>📷 EXIF Data ({exifKeys.length} tags)</h4>
                <JsonBlock obj={meta.exif} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------- Presenter Config ---------------- */
const DEFAULT_CFG = {
  presenter_title: "🐄 Breed Classifier — Demonstration",
  presenter_subtitle: "AI-Powered Cattle & Buffalo Breed Identification",
  confidence_display_mode: "scaled",
  confidence_floor_pct: 82.0,
  hide_species_confidence: true,
  hide_species_badge: false,
  hide_top5_list: false,
  top5_count: 3,
  min_breed_confidence_pct: 10.0,
  hide_info_footer: true,
  hide_model_selector: false,
  hide_batch_tab: false,
  hide_batch_summary_stats: true,
};

function PresenterConfig() {
  const [cfg, setCfg] = useState(DEFAULT_CFG);
  const [status, setStatus] = useState("");

  useEffect(() => {
    getPresenterConfig().then((c) => setCfg({ ...DEFAULT_CFG, ...c })).catch(() => {});
  }, []);

  const set = (k, v) => setCfg((c) => ({ ...c, [k]: v }));

  const save = async () => {
    try {
      await savePresenterConfig(cfg);
      setStatus("✅ Saved");
    } catch (e) {
      setStatus("❌ " + e.message);
    }
    setTimeout(() => setStatus(""), 3000);
  };

  const reset = async () => {
    setCfg(DEFAULT_CFG);
    try {
      await savePresenterConfig(DEFAULT_CFG);
      setStatus("↺ Reset to defaults");
    } catch (e) {
      setStatus("❌ " + e.message);
    }
    setTimeout(() => setStatus(""), 3000);
  };

  const Toggle = ({ k, label }) => (
    <div className="config-row">
      <label className="toggle-label">
        <input type="checkbox" checked={!!cfg[k]} onChange={(e) => set(k, e.target.checked)} />
        <span>{label}</span>
      </label>
    </div>
  );

  return (
    <div className="card">
      <h2>🎛️ Presenter View Customizer</h2>
      <p className="muted">Configures how the Python <code>--present</code> mode behaves. Saved to <code>outputs/logs/presenter_config.json</code>.</p>
      <div className="dev-panel">
        <h3 className="dev-h">🏷️ Branding & Titles</h3>
        <div className="grid-2">
          <div>
            <label className="field-label">Presenter Title</label>
            <input className="dev-input" type="text" value={cfg.presenter_title}
              onChange={(e) => set("presenter_title", e.target.value)} />
          </div>
          <div>
            <label className="field-label">Presenter Subtitle</label>
            <input className="dev-input" type="text" value={cfg.presenter_subtitle}
              onChange={(e) => set("presenter_subtitle", e.target.value)} />
          </div>
        </div>

        <h3 className="dev-h">🛡️ Confidence Presentation</h3>
        <div className="grid-2">
          <div>
            <label className="field-label">Breed Confidence Display Mode</label>
            <select className="dev-input" value={cfg.confidence_display_mode}
              onChange={(e) => set("confidence_display_mode", e.target.value)}>
              <option value="scaled">⚡ Normalized / Scaled</option>
              <option value="clamped">🛡️ Floor Minimum %</option>
              <option value="badge_only">🏷️ Qualitative Badge Only</option>
              <option value="hide">🚫 Hide Percentage</option>
              <option value="percentage">📊 Raw Exact Percentage</option>
            </select>
          </div>
          <div>
            <label className="field-label">Confidence Floor ({cfg.confidence_floor_pct}%)</label>
            <input type="range" min="50" max="95" step="1" value={cfg.confidence_floor_pct}
              onChange={(e) => set("confidence_floor_pct", parseFloat(e.target.value))} />
          </div>
        </div>

        <Toggle k="hide_species_confidence" label="Hide species confidence percentage" />
        <Toggle k="hide_species_badge" label="Hide species badge completely" />

        <h3 className="dev-h">👁️ Section Visibility</h3>
        <Toggle k="hide_top5_list" label="Hide Top Predictions list entirely" />
        <div className="grid-2">
          <div>
            <label className="field-label">Number of Top Breeds to Show</label>
            <select className="dev-input" value={cfg.top5_count}
              onChange={(e) => set("top5_count", parseInt(e.target.value))}>
              <option value={1}>1 (Top Breed only)</option>
              <option value={2}>2 (Top 2)</option>
              <option value={3}>3 (Top 3)</option>
              <option value={5}>5 (Top 5)</option>
            </select>
          </div>
          <div>
            <label className="field-label">Min breed confidence filter ({cfg.min_breed_confidence_pct}%)</label>
            <input type="range" min="0" max="40" step="1" value={cfg.min_breed_confidence_pct}
              onChange={(e) => set("min_breed_confidence_pct", parseFloat(e.target.value))} />
          </div>
        </div>
        <Toggle k="hide_info_footer" label="Hide model metadata footer" />
        <Toggle k="hide_model_selector" label="Hide Model Selector dropdown" />
        <Toggle k="hide_batch_tab" label="Hide Batch Upload tab" />
        <Toggle k="hide_batch_summary_stats" label="Hide batch summary statistics" />

        <div className="btn-row">
          <button className="btn primary" onClick={save}>💾 Save Config</button>
          <button className="btn ghost" onClick={reset}>↺ Reset Defaults</button>
          {status && <span style={{ color: "var(--green)", fontWeight: 600 }}>{status}</span>}
        </div>
      </div>
    </div>
  );
}

/* ---------------- Session Logs ---------------- */
function SessionLogs() {
  const [logs, setLogs] = useState("(loading…)");
  const load = () => getLogs().then(setLogs).catch((e) => setLogs("Failed: " + e.message));
  useEffect(() => { load(); }, []);
  return (
    <div className="card">
      <h2>📜 Session Log</h2>
      <button className="btn ghost" onClick={load}>🔄 Refresh Logs</button>
      <div className="dev-panel log-panel">
        <pre className="log-pre">{logs}</pre>
      </div>
    </div>
  );
}

export default function DevTools({ models }) {
  return (
    <div className="dev-tools-grid">
      <ModelSpec models={models} />
      <ClassMaps />
      <ImageMeta />
      <PresenterConfig />
      <SessionLogs />
    </div>
  );
}
