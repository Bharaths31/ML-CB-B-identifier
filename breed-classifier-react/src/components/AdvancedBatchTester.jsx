import { useRef, useState } from "react";
import { predict } from "../api.js";
import { useI18n } from "../i18n/I18nProvider.jsx";
import {
  formatBytes,
  optimizeImageTo260,
  releaseOptimizerResult,
} from "../utils/imageOptimizer.js";
import {
  addSessionEntry,
  makeSessionEntry,
  readSession,
} from "../utils/sessionStore.js";

/**
 * Batch tester inside the Advanced panel. Uses the currently selected model
 * and the Advanced fit configuration (stretch by default).
 */
export default function AdvancedBatchTester({ fit, model }) {
  const { t } = useI18n();
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0, name: "" });
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);
  const [sessionMsg, setSessionMsg] = useState(null);
  const inputRef = useRef(null);

  const pick = (event) => {
    const next = Array.from(event.target.files || []).filter((file) =>
      file.type.startsWith("image/")
    );
    setFiles(next);
    setResults([]);
    setSummary(null);
    setError(null);
    setSessionMsg(null);
    event.target.value = "";
  };

  const run = async () => {
    if (!files.length || busy) return;
    setBusy(true);
    setError(null);
    setResults([]);
    setSessionMsg(null);
    const out = [];
    let originalBytes = 0;
    let optimizedBytes = 0;
    let optimizedCount = 0;

    for (let i = 0; i < files.length; i += 1) {
      const source = files[i];
      let optimized = null;
      setProgress({ done: i, total: files.length, name: source.name });
      try {
        optimized = await optimizeImageTo260(source, {
          ...fit,
          allowUpscale: false,
        });
        originalBytes += source.size || 0;
        optimizedBytes += optimized.meta.optimizedBytes || 0;
        optimizedCount += 1;
        const data = await predict(optimized.file, model, source.name);
        data.filename = source.name;
        data.optimizerMeta = optimized.meta;
        out.push(data);
      } catch (err) {
        out.push({ error: err.message, filename: source.name });
      } finally {
        if (optimized) releaseOptimizerResult(optimized);
      }
      setResults([...out]);
    }

    const valid = out.filter((item) => !item.error);
    setSummary({
      total: out.length,
      valid: valid.length,
      cattle: valid.filter((item) => item.species === "Cattle").length,
      buffalo: valid.filter((item) => item.species === "Buffalo").length,
      ood: valid.filter((item) => item.is_ood).length,
      avgConfidence: valid.length
        ? valid.reduce((sum, item) => sum + (item.top_breed_confidence || 0), 0) / valid.length
        : 0,
      originalBytes,
      optimizedBytes,
      optimizedCount,
    });
    setProgress({ done: files.length, total: files.length, name: "" });
    setBusy(false);
  };

  const addToSession = () => {
    const valid = results.filter((item) => !item.error);
    if (!valid.length) return;
    let entries = readSession();
    for (const item of valid) {
      entries = addSessionEntry(
        makeSessionEntry({ result: item, quality: null, thumbnail: null, filename: item.filename }),
        entries
      );
    }
    setSessionMsg(t("advanced.addedToSession", { count: valid.length }));
  };

  return (
    <div className="advanced-batch">
      <div className="advanced-batch-head">
        <div>
          <h4>{t("advanced.batchTitle")}</h4>
          <p>{t("advanced.batchHint")}</p>
        </div>
        <span className="advanced-config-chip">
          {fit.fitMode === "stretch" ? t("optimizer.stretch") : t("optimizer.padMean")}
          {" · "}
          {model || t("header.defaultModel")}
        </span>
      </div>

      <div className="advanced-batch-actions">
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          style={{ display: "none" }}
          onChange={pick}
        />
        <button type="button" className="btn btn-outline" onClick={() => inputRef.current?.click()}>
          {t("advanced.chooseFiles")}
        </button>
        <button
          type="button"
          className="btn btn-primary"
          onClick={run}
          disabled={!files.length || busy}
        >
          {busy ? t("advanced.running") : t("advanced.runBatch")}
        </button>
      </div>

      {files.length > 0 && (
        <p className="advanced-batch-files">
          {files.map((file) => file.name).join(", ")}
        </p>
      )}

      {busy && (
        <p className="advanced-batch-progress">
          {progress.done + 1} / {progress.total}: {progress.name}
        </p>
      )}

      {error && <p className="camera-error-inline">{error}</p>}

      {summary && (
        <div className="advanced-batch-summary">
          <div className="advanced-summary-stats">
            <div><span>{t("advanced.images")}</span><strong>{summary.valid}/{summary.total}</strong></div>
            <div><span>{t("session.cattle")}</span><strong>{summary.cattle}</strong></div>
            <div><span>{t("session.buffalo")}</span><strong>{summary.buffalo}</strong></div>
            <div><span>{t("session.flagged")}</span><strong>{summary.ood}</strong></div>
            <div><span>{t("session.avgConfidence")}</span><strong>{Math.round(summary.avgConfidence)}%</strong></div>
            <div>
              <span>{t("advanced.bandwidth")}</span>
              <strong>
                {formatBytes(summary.originalBytes)} → {formatBytes(summary.optimizedBytes)}
              </strong>
            </div>
          </div>

          <div className="advanced-batch-results">
            {results.map((item, index) => (
              <div
                key={`${item.filename || "row"}-${index}`}
                className={`advanced-batch-row ${item.error ? "is-error" : ""} ${item.is_ood ? "is-ood" : ""}`}
              >
                <span className="advanced-row-file">{item.filename}</span>
                {item.error ? (
                  <span className="advanced-row-error">{item.error}</span>
                ) : (
                  <>
                    <span className="advanced-row-breed">{item.top_breed}</span>
                    <span className="advanced-row-conf">
                      {Number(item.top_breed_confidence || 0).toFixed(1)}%
                    </span>
                    {item.is_ood && <span className="advanced-row-ood">OOD</span>}
                  </>
                )}
              </div>
            ))}
          </div>

          <button type="button" className="btn btn-secondary btn-sm" onClick={addToSession}>
            {t("advanced.addToSession")}
          </button>
          {sessionMsg && <span className="advanced-session-msg">{sessionMsg}</span>}
        </div>
      )}
    </div>
  );
}
