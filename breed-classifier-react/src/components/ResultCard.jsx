import { useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { shareTraits } from "../data/breedTraits.js";
import { loadImage, renderShareCard, shareOrDownloadCanvas } from "../utils/cropAndQuality.js";
import {
  confidenceMeta,
  formatBreedConfidence,
  resolveConfidenceLevel,
} from "../utils/confidence.js";
import { normalizeTop } from "../utils/resultHelpers.js";
import { statusMeta } from "./QualityBadge.jsx";

/**
 * Compact main-view result. Detailed diagnostics, X-Ray, traits, quality
 * subscores, BPA registration and vet escalation live in the Advanced drawer.
 */
export default function ResultCard({
  result,
  config,
  onReset,
  onRetake,
  quality,
  previewUrl,
  onOpenAdvanced,
}) {
  const { t } = useI18n();
  const [sharing, setSharing] = useState(false);

  if (!result) return null;

  const isOod = Boolean(result.is_ood);
  const species = result.predicted_species || result.species || "Cattle";
  const breed = (result.predicted_breed || result.top_breed || result.breed || "Unknown").replace(/_/g, " ");
  const confidence = typeof result.confidence === "number" ? result.confidence : 0;
  const top = normalizeTop(result);
  const topProb = top[0]?.prob ?? confidence;

  const level = resolveConfidenceLevel(result);
  const levelMeta = confidenceMeta(level);

  const maxBreeds = config?.display?.max_breeds ?? (config?.display?.top_k ?? 5);
  const minConf = config?.display?.min_breed_confidence_pct ?? 0;
  const displayedTop = top
    .filter((b) => b.prob * 100 >= minConf)
    .slice(0, Math.max(1, Math.min(3, maxBreeds)));

  const confidenceMode = config?.display?.confidence_mode || "percentage";
  const confidenceFloor = config?.display?.confidence_floor_pct ?? 80;
  const showSpeciesConfidence = config?.display?.show_species_confidence ?? true;
  const showSpeciesBadge = config?.display?.show_species_badge ?? true;
  const displayConfidence = formatBreedConfidence(
    topProb * 100,
    top.slice(0, 3).map((item) => item.prob * 100),
    confidenceMode,
    confidenceFloor
  );
  const qualityMeta = quality ? statusMeta(quality.status) : null;

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

  return (
    <div className="result-card-container result-card-compact">
      <div className={`prediction-main-card compact-prediction-card ${isOod ? "is-ood" : ""}`}>
        <div className="compact-result-head">
          <div className="compact-result-titles">
            <div className="species-badge-row">
              {(showSpeciesBadge || isOod) && (
                <span className={`species-pill species-${isOod ? "unknown" : String(species).toLowerCase()}`}>
                  {isOod ? t("result.oodBadge") : String(species).toUpperCase()}
                </span>
              )}
              <span
                className={`confidence-pill confidence-pill-level-${level}`}
                title={t(levelMeta.key)}
              >
                {t("confidence.levelShort", { level })}
              </span>
              {showSpeciesConfidence && !isOod && (
                <span className="species-confidence-pill">
                  {t("result.speciesConfidence")}: {Math.round(Number(result.species_confidence) || 0)}%
                </span>
              )}
              {quality && (
                <span className={`quality-compact ${quality.status === "READY" ? "quality-good" : "quality-warn"}`}>
                  <span aria-hidden="true">{qualityMeta?.icon}</span>
                  {quality.overallScore}
                </span>
              )}
            </div>

            <h2 className="predicted-breed-title">
              {isOod ? t("result.oodTitle") : breed}
            </h2>
            {!isOod && (
              <span className="compact-confidence-text">
                {displayConfidence.hidden
                  ? displayConfidence.value
                  : `${displayConfidence.value} ${t("result.confidence")}`}
              </span>
            )}
          </div>

          <button
            type="button"
            className="btn btn-outline btn-sm compact-details-btn"
            onClick={onOpenAdvanced}
          >
            ⚙ {t("advanced.openDetails")}
          </button>
        </div>

        {isOod ? (
          <div className="ood-advisory-box compact-ood-box">
            <span>{t("result.oodBody")}</span>
          </div>
        ) : (
          <>
            {!displayConfidence.hidden && (
              <div className="confidence-meter-container">
                <div className="confidence-bar-bg">
                  <div
                    className={`confidence-bar-fill ${level <= 2 ? "bar-warning" : level === 3 ? "bar-medium" : "bar-success"}`}
                    style={{ width: `${Math.min(100, Math.max(0, Math.round(topProb * 100)))}%` }}
                  />
                </div>
                <div className="confidence-scale-labels">
                  <span>0%</span>
                  <span>50%</span>
                  <span>100%</span>
                </div>
              </div>
            )}

            {displayedTop.length > 0 && (
              <div className="compact-top3">
                {displayedTop.map((item, idx) => (
                  <div key={`${item.name}-${idx}`} className="compact-top3-row">
                    <span className="compact-top3-rank">#{idx + 1}</span>
                    <span className="compact-top3-name">{item.name}</span>
                    <span className="compact-top3-pct">{Math.round(item.prob * 100)}%</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        <div className="card-footer-actions compact-result-actions">
          <button type="button" className="btn btn-outline" onClick={handleShare} disabled={sharing}>
            {sharing ? "…" : t("actions.download")}
          </button>
          {onRetake && !isOod && (
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
