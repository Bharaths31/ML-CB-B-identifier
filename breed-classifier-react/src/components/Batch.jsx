import { useRef, useState } from "react";
import ModelSelect from "./ModelSelect.jsx";
import PredictionProgress from "./PredictionProgress.jsx";
import { predict, exportOdt, downloadBlob } from "../api.js";
import {
  formatBytes,
  optimizeImageTo260,
  releaseOptimizerResult,
} from "../utils/imageOptimizer.js";
import {
  SESSION_LIMIT,
  makeSessionEntry,
  readSession,
  writeSession,
} from "../utils/sessionStore.js";

export default function Batch({ models, model, setModel, devMode, predictionEvent }) {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0, name: "", phase: "" });
  const [exporting, setExporting] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [optimization, setOptimization] = useState(null);
  const inputRef = useRef(null);

  const pick = (fl) => {
    const imgs = Array.from(fl).filter((f) => f.type.startsWith("image/"));
    if (imgs.length) {
      setFiles(imgs);
      setResults([]);
      setOptimization(null);
    }
  };

  const run = async () => {
    if (!files.length) return;
    setBusy(true);
    const out = [];
    let originalBytes = 0;
    let optimizedBytes = 0;
    let optimizedCount = 0;

    for (let i = 0; i < files.length; i++) {
      const source = files[i];
      let optimized = null;
      setProgress({
        done: i,
        total: files.length,
        name: source.name,
        phase: "optimizing",
      });
      try {
        optimized = await optimizeImageTo260(source, {
          fitMode: "pad",
          padStyle: "imagenet_mean",
          allowUpscale: false,
        });
        originalBytes += source.size || 0;
        optimizedBytes += optimized.meta.optimizedBytes || 0;
        optimizedCount += 1;

        setProgress({
          done: i,
          total: files.length,
          name: source.name,
          phase: "predicting",
        });
        const data = await predict(optimized.file, model, source.name);
        data.filename = source.name;
        data.optimizerMeta = optimized.meta;
        out.push(data);
      } catch (e) {
        out.push({ error: e.message, filename: source.name });
      } finally {
        if (optimized) releaseOptimizerResult(optimized);
      }
      setResults([...out]);
    }

    if (optimizedCount) {
      setOptimization({
        count: optimizedCount,
        originalBytes,
        optimizedBytes,
        savingsPct:
          originalBytes > 0
            ? Math.round((1 - optimizedBytes / originalBytes) * 1000) / 10
            : 0,
      });
    }
    setProgress({ done: files.length, total: files.length, name: "", phase: "" });
    setBusy(false);
  };

  const clear = () => {
    setFiles([]);
    setResults([]);
    setOptimization(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const doExport = async () => {
    setExporting(true);
    try {
      const blob = await exportOdt(results, "batch");
      downloadBlob(blob, `breed_batch_result_${Date.now()}.odt`);
    } catch (e) {
      alert(e.message);
    } finally {
      setExporting(false);
    }
  };

  const addToSession = () => {
    const validResults = results.filter((r) => !r.error);
    const additions = validResults.map((r) =>
      makeSessionEntry({ result: r, quality: null, thumbnail: null, filename: r.filename })
    );
    const next = [...additions, ...readSession()].slice(0, SESSION_LIMIT);
    writeSession(next);
    alert(`${validResults.length} animals added to the herd session.`);
  };

  const valid = results.filter((r) => !r.error);
  const cattle = valid.filter((r) => r.species === "Cattle").length;
  const buffalo = valid.filter((r) => r.species === "Buffalo").length;
  const avgConf = valid.length
    ? (valid.reduce((s, r) => s + (r.top_breed_confidence || 0), 0) / valid.length).toFixed(1)
    : "0.0";

  return (
    <div className="container">
      <div className="card">
        <h2>📁 Batch Upload</h2>
        {!files.length ? (
          <div
            className={"dropzone" + (dragOver ? " drag-over" : "")}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); pick(e.dataTransfer.files); }}
          >
            <span className="icon">📂</span>
            <p>Drag & drop multiple images here<br />or click to browse</p>
            <div className="formats">Supports: PNG, JPG, JPEG, BMP, WebP</div>
            <input
              ref={inputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.webp"
              multiple
              hidden
              onChange={(e) => pick(e.target.files)}
            />
          </div>
        ) : (
          <>
            <div className="batch-file-list">
              {files.map((f) => (
                <div className="batch-file-item" key={f.name}>
                  <span className="bf-name">📄 {f.name}</span>
                  <span className="bf-size">{(f.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
              ))}
            </div>
            <button className="btn ghost full" onClick={clear} disabled={busy}>🗑️ Clear All</button>
          </>
        )}

        <ModelSelect models={models} value={model} onChange={setModel} />

        <button className="btn primary full" onClick={run} disabled={!files.length || busy}>
          {busy ? "⏳ Analyzing…" : "🔍 Analyze All Images"}
        </button>

        <PredictionProgress predictionEvent={predictionEvent} isBusy={busy} />

        {busy && (
          <>
            <div className="progress-wrap">
              <div
                className="progress-bar"
                style={{ width: `${(progress.done / Math.max(progress.total, 1)) * 100}%` }}
              />
            </div>
            <div className="progress-text">
              {progress.done + 1} / {progress.total}: {progress.name}
              {progress.phase === "optimizing" ? " · Optimizing to 260×260…" : " · Predicting…"}
            </div>
          </>
        )}
      </div>

      <div className="card">
        <h2>📊 Batch Results</h2>
        {!results.length ? (
          <div className="placeholder">
            <span className="placeholder-icon">📊</span>
            <p>Select images and click <strong>Analyze All</strong> to see results</p>
          </div>
        ) : (
          <>
            {optimization && (
              <div className="optimizer-bar optimizer-bar-compact">
                <span className="optimizer-badge">
                  Optimized {optimization.count} images:{" "}
                  {formatBytes(optimization.originalBytes)} →{" "}
                  {formatBytes(optimization.optimizedBytes)}
                  {optimization.savingsPct
                    ? ` (${optimization.savingsPct > 0 ? "-" : "+"}${Math.abs(optimization.savingsPct)}%)`
                    : ""}
                </span>
              </div>
            )}

            <div className="batch-summary">
              <div>
                <div className="stat-val">{valid.length}</div>
                <div className="stat-label">Images</div>
              </div>
              <div>
                <div className="stat-val">{cattle} / {buffalo}</div>
                <div className="stat-label">Cattle / Buffalo</div>
              </div>
              <div>
                <div className="stat-val">{avgConf}%</div>
                <div className="stat-label">Avg Confidence</div>
              </div>
            </div>

            <div className="batch-action-row">
              <button className="btn secondary full" onClick={addToSession}>
                ＋ Add All to Herd Session
              </button>
              {devMode && (
                <button className="btn outline-green full" onClick={doExport} disabled={exporting}>
                  {exporting ? "⏳ Generating ODT…" : "📄 Export All to ODT"}
                </button>
              )}
            </div>

            <div className="batch-results-wrap">
              {results.map((r, i) =>
                r.error ? (
                  <div className="batch-card" key={i}>
                    <div className="bc-header">
                      <span className="bc-filename">📄 {r.filename}</span>
                    </div>
                    <div style={{ color: "var(--red)" }}>❌ {r.error}</div>
                  </div>
                ) : (
                  <div className={"batch-card" + (r.is_ood ? " batch-card-ood" : "")} key={i}>
                    <div className="bc-header">
                      <span className="bc-filename">📄 {r.filename}</span>
                      <span className={"bc-species " + String(r.species).toLowerCase()}>
                        {r.species} {devMode && r.species_confidence.toFixed(1) + "%"}
                      </span>
                      {r.is_ood && <span className="bc-ood">OOD</span>}
                    </div>
                    <div className="bc-breed">{r.top_breed}</div>
                    <div className="bc-conf">{r.top_breed_confidence.toFixed(1)}% confidence</div>
                    <ul className="bc-top5">
                      {r.top5_breeds.map((b) => (
                        <li key={b.breed}>
                          <span>{b.breed}</span>
                          <span>{b.confidence.toFixed(1)}%</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
