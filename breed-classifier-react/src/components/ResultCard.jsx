import { useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { shareTraits } from "../data/breedTraits.js";
import { loadImage, renderShareCard, shareOrDownloadCanvas } from "../utils/cropAndQuality.js";
import { formatBytes } from "../utils/imageOptimizer.js";
import { statusMeta } from "./QualityBadge.jsx";
import TraitComparison from "./TraitComparison.jsx";

function normalizeTop(result) {
  const raw = result?.top_k || result?.top5_breeds || [];
  const seen = new Set();
  const out = [];
  for (const item of raw) {
    const name = (item.breed || item.name || "").replace(/_/g, " ");
    if (!name || seen.has(name.toLowerCase())) continue;
    seen.add(name.toLowerCase());
    const prob =
      typeof item.prob === "number"
        ? item.prob
        : typeof item.confidence === "number"
        ? item.confidence / 100
        : 0;
    out.push({ name, prob });
  }
  return out;
}

function oodTriggers(details) {
  if (!details) return [];
  const triggers = [];
  if (details.ood_by_energy) triggers.push("energy");
  if (details.ood_by_msp) triggers.push("msp");
  if (details.ood_by_binary_floor) triggers.push("floor");
  return triggers;
}

export default function ResultCard({
  result,
  config,
  onReset,
  onRetake,
  quality,
  previewUrl,
  optimizerMeta,
}) {
  const { t } = useI18n();
  const [showDetails, setShowDetails] = useState(
    config?.display?.show_technical_details ?? false
  );
  const [sharing, setSharing] = useState(false);

  if (!result) return null;

  const isOod = Boolean(result.is_ood);
  const species = result.predicted_species || result.species || "Cattle";
  const breed = (result.predicted_breed || result.top_breed || result.breed || "Unknown").replace(/_/g, " ");
  const confidence = typeof result.confidence === "number" ? result.confidence : 0;
  const top = normalizeTop(result);
  const topProb = top[0]?.prob ?? confidence;
  const margin = top.length > 1 ? top[0].prob - top[1].prob : topProb;

  const tier = isOod ? 3 : topProb >= 0.72 && margin >= 0.12 ? 1 : 2;
  const tierMeta = {
    1: { cls: "tier-high", icon: "✓", title: t("result.tier1"), body: t("result.tier1Body") },
    2: { cls: "tier-medium", icon: "≈", title: t("result.tier2"), body: t("result.tier2Body") },
    3: { cls: "tier-low", icon: "!", title: t("result.tier3"), body: t("result.tier3Body") },
  }[tier];

  const details = result.ood_details || {};
  const triggers = oodTriggers(details);
  const maxBreeds = config?.display?.max_breeds ?? (config?.display?.top_k ?? 5);
  const minConf = config?.display?.min_breed_confidence_pct ?? 0;
  const displayedTop = top
    .filter((b) => b.prob * 100 >= minConf)
    .slice(0, Math.max(1, maxBreeds));

  const handleShare = async () => {
    setSharing(true);
    try {
      const canvas = document.createElement("canvas");
      let image = null;
      if (previewUrl) {
        try {
          image = await loadImage(previewUrl);
        } catch {
          image = null;
        }
      }
      renderShareCard(canvas, {
        image,
        result: { ...result, predicted_breed: breed, predicted_species: species, confidence: topProb },
        quality,
        traits: shareTraits(top[0]?.name || breed),
      });
      await shareOrDownloadCanvas(canvas, `breed_${breed.replace(/\s+/g, "_")}_${Date.now()}.png`);
    } catch (err) {
      alert(err?.message || t("result.shareError"));
    } finally {
      setSharing(false);
    }
  };

  const qualityMeta = quality ? statusMeta(quality.status) : null;

  return (
    <div className="result-card-container">
      {/* Tier banner */}
      <div className={`tier-banner ${tierMeta.cls}`} role="status">
        <span className="tier-icon" aria-hidden="true">{tierMeta.icon}</span>
        <div className="tier-copy">
          <strong>{tierMeta.title}</strong>
          <span>{tierMeta.body}</span>
        </div>
        {tier !== 1 && onRetake && (
          <button type="button" className="btn btn-sm tier-retake" onClick={onRetake}>
            {t("result.retakeCoach")}
          </button>
        )}
      </div>

      {/* OOD trigger diagnostics */}
      {tier === 3 && (
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

      {/* Main prediction header */}
      <div className="prediction-main-card">
        <div className="prediction-header">
          <div className="species-badge-row">
            <span className={`species-pill species-${tier === 3 ? "unknown" : String(species).toLowerCase()}`}>
              {tier === 3 ? t("result.oodBadge") : String(species).toUpperCase()}
            </span>
            <span className="confidence-pill">
              {tier === 3 ? t("result.tier3") : `${Math.round(topProb * 100)}% ${t("result.confidence")}`}
            </span>
            {quality && (
              <span className={`quality-compact ${quality.status === "READY" ? "quality-good" : "quality-warn"}`}>
                <span aria-hidden="true">{qualityMeta?.icon}</span>
                {quality.overallScore}
              </span>
            )}
          </div>

          <h2 className="predicted-breed-title">
            {tier === 3 ? t("result.oodTitle") : breed}
          </h2>

          <div className="confidence-meter-container">
            <div className="confidence-bar-bg">
              <div
                className={`confidence-bar-fill ${tier === 3 ? "bar-warning" : tier === 2 ? "bar-medium" : "bar-success"}`}
                style={{ width: `${tier === 3 ? 15 : Math.min(100, Math.max(0, Math.round(topProb * 100)))}%` }}
              />
            </div>
            <div className="confidence-scale-labels">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>
        </div>

        {/* Ranked list / withheld */}
        {tier === 3 ? (
          <div className="top-matches-section">
            <h3 className="section-subtitle">{t("result.withheld")}</h3>
            <div className="ood-advisory-box">
              <span>{t("result.oodBody")}</span>
            </div>
          </div>
        ) : (
          displayedTop.length > 0 && (
            <div className="top-matches-section">
              <h3 className="section-subtitle">{t("result.ranked")}</h3>
              <div className="breed-list">
                {displayedTop.map((item, idx) => {
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
            </div>
          )
        )}

        {/* Why this breed + look-alike matrix */}
        {tier !== 3 && top.length > 0 && (
          <TraitComparison
            breeds={top.slice(0, 3).map((b) => b.name)}
            defaultOpen={tier === 2}
          />
        )}

        {/* Quality report */}
        {quality && (
          <div className="result-quality-row">
            <span className="result-quality-label">{t("result.quality")}</span>
            <span className={`quality-compact ${quality.status === "READY" ? "quality-good" : "quality-warn"}`}>
              <span aria-hidden="true">{qualityMeta?.icon}</span>
              {quality.overallScore}/100
            </span>
            <span className="result-quality-note">{t(`quality.${qualityHintKey(quality.status)}`)}</span>
          </div>
        )}

        {/* Technical details */}
        <div className="diagnostics-toggle-row">
          <button
            type="button"
            className="btn-text-action"
            onClick={() => setShowDetails(!showDetails)}
          >
            <span aria-hidden="true">{showDetails ? "▾" : "▸"}</span>
            {showDetails ? t("result.technicalHide") : t("result.technical")}
          </button>
        </div>

        {showDetails && (
          <div className="diagnostics-panel">
            <div className="diagnostics-grid">
              <div className="diag-item">
                <span className="diag-label">{t("result.inference")}</span>
                <span className="diag-value">
                  {result.inference_time_ms ? `${result.inference_time_ms} ms` : t("result.notAvailable")}
                </span>
              </div>
              <div className="diag-item">
                <span className="diag-label">{t("result.modelName")}</span>
                <span className="diag-value text-truncate" title={result.model_name}>
                  {result.model_name || config?.active_model || t("header.defaultModel")}
                </span>
              </div>
              <div className="diag-item">
                <span className="diag-label">{t("result.filename")}</span>
                <span className="diag-value text-truncate" title={result.filename || ""}>
                  {result.filename || t("result.notAvailable")}
                </span>
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
                  {result.species_prob != null
                    ? `${Math.round(result.species_prob * 100)}%`
                    : t("result.notAvailable")}
                </span>
              </div>
              {optimizerMeta && (
                <div className="diag-item">
                  <span className="diag-label">{t("optimizer.modelInput")}</span>
                  <span className="diag-value">
                    260×260 · {formatBytes(optimizerMeta.optimizedBytes)} ·{" "}
                    {optimizerMeta.fitMode === "stretch"
                      ? t("optimizer.stretch")
                      : t("optimizer.zeroCrop")}
                  </span>
                </div>
              )}
              {quality && (
                <>
                  <div className="diag-item">
                    <span className="diag-label">{t("quality.sharpness")}</span>
                    <span className="diag-value">{quality.sharpnessScore}/100</span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">{t("quality.lighting")}</span>
                    <span className="diag-value">{quality.exposureScore}/100</span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">{t("quality.framing")}</span>
                    <span className="diag-value">{quality.framingScore}/100</span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        <div className="card-footer-actions card-footer-multi">
          <button type="button" className="btn btn-outline" onClick={handleShare} disabled={sharing}>
            {sharing ? "…" : t("actions.download")}
          </button>
          {onRetake && tier !== 3 && (
            <button type="button" className="btn btn-secondary" onClick={onRetake}>
              {t("result.retakeCoach")}
            </button>
          )}
          <button type="button" className="btn btn-primary" onClick={onReset}>
            {t("result.classifyAnother")}
          </button>
        </div>
      </div>
    </div>
  );
}

function qualityHintKey(status) {
  switch (status) {
    case "TOO_BLURRY":
      return "tooBlurry";
    case "TOO_DARK":
      return "tooDark";
    case "OVEREXPOSED":
      return "overexposed";
    case "NO_SUBJECT_OR_TOO_FAR":
      return "noSubject";
    case "PARTIALLY_OUT_OF_FRAME":
      return "partial";
    default:
      return "ready";
  }
}
