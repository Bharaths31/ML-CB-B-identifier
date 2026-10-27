import { useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { confidenceMeta, resolveConfidenceLevel } from "../utils/confidence.js";
import { formatBytes } from "../utils/imageOptimizer.js";
import { normalizeTop, oodTriggers, qualityHintKey } from "../utils/resultHelpers.js";
import { getBrowserGps } from "../utils/escalationQueue.js";
import { statusMeta } from "./QualityBadge.jsx";
import TraitComparison from "./TraitComparison.jsx";
import BpaRegistrationModal from "./BpaRegistrationModal.jsx";

/**
 * Detailed prediction diagnostics moved out of the main user card.
 * Rendered inside the header Advanced drawer.
 */
export default function AdvancedResultDetails({
  result,
  config,
  quality,
  previewUrl,
  modelInputUrl,
  optimizerMeta,
  onEscalate,
  onOpenExpert,
}) {
  const { t } = useI18n();
  const [showDetails, setShowDetails] = useState(
    config?.display?.show_technical_details ?? false
  );
  const [xrayOn, setXrayOn] = useState(false);
  const [xrayOpacity, setXrayOpacity] = useState(65);
  const [escalating, setEscalating] = useState(false);
  const [escalated, setEscalated] = useState(false);
  const [bpaOpen, setBpaOpen] = useState(false);
  const [gpsFix, setGpsFix] = useState(null);

  if (!result) return null;

  const isOod = Boolean(result.is_ood);
  const species = result.predicted_species || result.species || "Cattle";
  const breed = (result.predicted_breed || result.top_breed || result.breed || "Unknown").replace(/_/g, " ");
  const top = normalizeTop(result);
  const topProb = top[0]?.prob ?? (Number(result.confidence) || 0);
  const level = resolveConfidenceLevel(result);
  const levelMeta = confidenceMeta(level);
  const needEscalation = level <= 2;
  const showEscalationWarning = config?.display?.show_escalation_warning ?? true;
  const details = result.ood_details || {};
  const triggers = oodTriggers(details);
  const attentionMap = result.attention_map_b64;
  const spatialMap = result.spatial_attention_b64;
  const modelInput = modelInputUrl || previewUrl;
  const qualityMeta = quality ? statusMeta(quality.status) : null;

  const handleEscalate = async () => {
    if (!onEscalate) return;
    setEscalating(true);
    try {
      await onEscalate(result, { manual: !needEscalation });
      setEscalated(true);
    } finally {
      setEscalating(false);
    }
  };

  const handleOpenBpa = async () => {
    setBpaOpen(true);
    try {
      setGpsFix(await getBrowserGps());
    } catch {
      setGpsFix(null);
    }
  };

  return (
    <div className="advanced-result-details">
      <div className="advanced-section-head">
        <div>
          <h3>{t("advanced.resultTitle")}</h3>
          <p>{t("advanced.resultSubtitle")}</p>
        </div>
      </div>

      {showEscalationWarning && (
        <div
          className={`confidence-level-banner confidence-level-${level}`}
          style={{ "--level-colour": levelMeta.badge }}
          role="status"
        >
          <span className="confidence-level-number" aria-hidden="true">{level}</span>
          <div className="tier-copy">
            <strong>
              {t(levelMeta.key)} · {Math.round(topProb * 100)}%
            </strong>
            <span>{t(levelMeta.actionKey)}</span>
          </div>
          {needEscalation && (
            <span className="confidence-level-trigger">{t("confidence.escalationTriggered")}</span>
          )}
        </div>
      )}

      {isOod && (
        <div className="ood-trigger-box">
          <h4>{t("result.trigger")}</h4>
          <ul>
            {triggers.length === 0 && <li>{t("result.tier3Body")}</li>}
            {triggers.map((key) => (
              <li key={key}>
                {t(`result.trigger.${key}`)}
                {details.energy != null && key === "energy" && (
                  <span className="ood-trigger-value">
                    {" "}
                    {Number(details.energy).toFixed(3)} &gt; {Number(details.energy_threshold).toFixed(3)}
                  </span>
                )}
                {details.msp != null && key !== "energy" && (
                  <span className="ood-trigger-value">
                    {" "}
                    {Number(details.msp).toFixed(3)} &lt;{" "}
                    {Number(key === "msp" ? details.msp_threshold : details.binary_confidence_floor).toFixed(3)}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {!isOod && top.length > 0 && (
        <section className="advanced-result-block">
          <h4>{t("result.ranked")}</h4>
          <div className="breed-list">
            {top.slice(0, 5).map((item, idx) => {
              const pct = Math.round(item.prob * 100);
              return (
                <div key={`${item.name}-${idx}`} className={`breed-row ${idx === 0 ? "breed-row-top" : ""}`}>
                  <div className="breed-row-header">
                    <span className="breed-rank">#{idx + 1}</span>
                    <span className="breed-name">{item.name}</span>
                    <span className="breed-pct">{pct}%</span>
                  </div>
                  <div className="breed-mini-bar-bg">
                    <div className="breed-mini-bar-fill" style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {modelInput && (
        <section className="advanced-result-block">
          <div className="xray-head">
            <div>
              <h4 className="section-subtitle">{t("xray.title")}</h4>
              <p className="xray-subtitle">{t("xray.subtitle")}</p>
            </div>
            <button
              type="button"
              className={`btn btn-sm ${xrayOn ? "btn-primary" : "btn-outline"}`}
              onClick={() => setXrayOn((value) => !value)}
              disabled={!attentionMap}
              title={attentionMap ? "" : t("xray.unavailable")}
            >
              {xrayOn ? t("xray.hide") : t("xray.show")}
            </button>
          </div>
          <div className="xray-body">
            <div className="xray-frame">
              <img src={modelInput} alt="260 by 260 model input" />
              {xrayOn && attentionMap && (
                <img
                  src={attentionMap}
                  alt="CBAM class activation heatmap"
                  className="xray-overlay"
                  style={{ opacity: xrayOpacity / 100 }}
                />
              )}
              <span className="preview-260-badge">260×260</span>
            </div>
            <div className="xray-controls">
              <label htmlFor="advanced-xray-opacity">{t("xray.opacity")}</label>
              <input
                id="advanced-xray-opacity"
                type="range"
                min="0"
                max="100"
                value={xrayOpacity}
                onChange={(event) => setXrayOpacity(Number(event.target.value))}
              />
              <div className="xray-legend">
                <span>{t("xray.low")}</span>
                <span className="xray-legend-bar" aria-hidden="true" />
                <span>{t("xray.high")}</span>
              </div>
              {spatialMap && xrayOn && (
                <div className="xray-spatial">
                  <span>{t("xray.cbamLayer")}</span>
                  <img src={spatialMap} alt="CBAM spatial attention mask" />
                </div>
              )}
              {result.attention_meta?.dominant_quadrant && (
                <p className="xray-meta">
                  {t("xray.focus")}: {result.attention_meta.dominant_quadrant.replace(/_/g, " ")}
                  {result.attention_meta.feature_grid
                    ? ` · F ${result.attention_meta.feature_grid.join("×")}`
                    : ""}
                  {result.attention_meta.cbam_grid
                    ? ` · CBAM ${result.attention_meta.cbam_grid.join("×")}`
                    : ""}
                </p>
              )}
              {!attentionMap && <p className="xray-meta">{t("xray.unavailable")}</p>}
            </div>
          </div>
        </section>
      )}

      {!isOod && top.length > 0 && (
        <section className="advanced-result-block">
          <TraitComparison
            breeds={top.slice(0, 3).map((item) => item.name)}
            defaultOpen={level <= 3}
          />
        </section>
      )}

      {quality && (
        <section className="advanced-result-block">
          <h4>{t("result.quality")}</h4>
          <div className="advanced-quality-grid">
            <div><span>{t("quality.score")}</span><strong>{quality.overallScore}/100</strong></div>
            <div><span>{t("quality.sharpness")}</span><strong>{quality.sharpnessScore}/100</strong></div>
            <div><span>{t("quality.lighting")}</span><strong>{quality.exposureScore}/100</strong></div>
            <div><span>{t("quality.framing")}</span><strong>{quality.framingScore}/100</strong></div>
          </div>
          <p className="advanced-result-note">{t(`quality.${qualityHintKey(quality.status)}`)}</p>
        </section>
      )}

      <section className="advanced-result-block">
        <button
          type="button"
          className="btn-text-action"
          onClick={() => setShowDetails(!showDetails)}
        >
          <span aria-hidden="true">{showDetails ? "▾" : "▸"}</span>
          {showDetails ? t("result.technicalHide") : t("result.technical")}
        </button>
        {showDetails && (
          <div className="diagnostics-panel">
            <div className="diagnostics-grid">
              <div className="diag-item">
                <span className="diag-label">{t("result.inference")}</span>
                <span className="diag-value">{result.inference_time_ms ? `${result.inference_time_ms} ms` : t("result.notAvailable")}</span>
              </div>
              <div className="diag-item">
                <span className="diag-label">{t("result.modelName")}</span>
                <span className="diag-value text-truncate" title={result.model_name}>
                  {result.model_name || config?.active_model || t("header.defaultModel")}
                </span>
              </div>
              <div className="diag-item">
                <span className="diag-label">{t("confidence.level")}</span>
                <span className="diag-value">{level} · {t(levelMeta.key)}</span>
              </div>
              <div className="diag-item">
                <span className="diag-label">{t("result.filename")}</span>
                <span className="diag-value text-truncate" title={result.filename || ""}>{result.filename || t("result.notAvailable")}</span>
              </div>
              <div className="diag-item">
                <span className="diag-label">{t("result.oodEval")}</span>
                <span className="diag-value">
                  {config?.ood?.enabled
                    ? isOod
                      ? t("result.oodFlagged")
                      : t("result.oodValid")
                    : t("result.oodDisabled")}
                </span>
              </div>
              <div className="diag-item">
                <span className="diag-label">{t("result.speciesProb")}</span>
                <span className="diag-value">
                  {result.species_prob != null ? `${Math.round(result.species_prob * 100)}%` : t("result.notAvailable")}
                </span>
              </div>
              {optimizerMeta && (
                <div className="diag-item">
                  <span className="diag-label">{t("optimizer.modelInput")}</span>
                  <span className="diag-value">
                    260×260 · {formatBytes(optimizerMeta.optimizedBytes)} · {optimizerMeta.fitMode === "stretch" ? t("optimizer.stretch") : t("optimizer.zeroCrop")}
                    {optimizerMeta.clahe ? " · CLAHE" : ""}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </section>

      <div className="advanced-result-actions">
        {showEscalationWarning && (
          <button
            type="button"
            className={`btn ${needEscalation ? "btn-danger" : "btn-outline"}`}
            onClick={handleEscalate}
            disabled={escalating || escalated || !onEscalate}
          >
            {escalated
              ? t("escalation.queued")
              : escalating
              ? t("escalation.sending")
              : needEscalation
              ? t("escalation.escalateTriggered")
              : t("escalation.escalateManual")}
          </button>
        )}
        {escalated && onOpenExpert && (
          <button type="button" className="btn btn-outline" onClick={onOpenExpert}>
            {t("escalation.openPortal")}
          </button>
        )}
        {level >= 3 && !isOod && (
          <button type="button" className="btn btn-primary" onClick={handleOpenBpa}>
            {t("bpa.register")}
          </button>
        )}
      </div>

      <BpaRegistrationModal
        open={bpaOpen}
        onClose={() => setBpaOpen(false)}
        result={{ ...result, confidence_level: level, predicted_species: species, predicted_breed: breed }}
        config={config}
        quality={quality}
        location={gpsFix || { location_label: "Manual capture" }}
        modelInputUrl={modelInput}
      />
    </div>
  );
}
