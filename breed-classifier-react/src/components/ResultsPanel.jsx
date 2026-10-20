import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import TraitComparison from "./TraitComparison.jsx";

/** Single-image prediction result: hero card + top-5 bars + footer. */
export default function ResultsPanel({ result, devMode, filename, onExport, exporting }) {
  const { t } = useI18n();
  const [bars, setBars] = useState(false);
  useEffect(() => {
    setBars(false);
    const t = requestAnimationFrame(() => requestAnimationFrame(() => setBars(true)));
    return () => cancelAnimationFrame(t);
  }, [result]);

  if (!result) {
    return (
      <div className="placeholder">
        <span className="placeholder-icon">🔬</span>
        <p>Upload or capture an image, then click <strong>Analyze Breed</strong></p>
      </div>
    );
  }

  if (result.error) {
    return (
      <div className="placeholder">
        <span className="placeholder-icon">❌</span>
        <p style={{ color: "var(--red)" }}>{result.error}</p>
      </div>
    );
  }

  const top5 = result.top5_breeds || [];
  const max = Math.max(...top5.map((b) => b.confidence), 1);
  const topProb = (result.confidence ?? (top5[0]?.confidence || 0) / 100) || 0;
  const margin = top5.length > 1 ? (top5[0].confidence - top5[1].confidence) / 100 : topProb;
  const tier = result.is_ood ? 3 : topProb >= 0.72 && margin >= 0.12 ? 1 : 2;
  const details = result.ood_details || {};

  return (
    <div className="results fade-up">
      <div className={`tier-banner ${tier === 1 ? "tier-high" : tier === 2 ? "tier-medium" : "tier-low"}`} role="status">
        <span className="tier-icon" aria-hidden="true">{tier === 1 ? "✓" : tier === 2 ? "≈" : "!"}</span>
        <div className="tier-copy">
          <strong>{t(`result.tier${tier}`)}</strong>
          <span>{t(`result.tier${tier}Body`)}</span>
        </div>
      </div>

      {tier === 3 && (
        <div className="ood-trigger-box">
          <h4>{t("result.trigger")}</h4>
          <ul>
            {details.ood_by_energy && <li>{t("result.trigger.energy")}</li>}
            {details.ood_by_msp && <li>{t("result.trigger.msp")}</li>}
            {details.ood_by_binary_floor && <li>{t("result.trigger.floor")}</li>}
            {!details.ood_by_energy && !details.ood_by_msp && !details.ood_by_binary_floor && (
              <li>{t("result.tier3Body")}</li>
            )}
          </ul>
        </div>
      )}

      <div className="result-hero">
        <div className={"species-badge " + result.species.toLowerCase()}>
          {result.species} {devMode && `${result.species_confidence.toFixed(1)}%`}
        </div>
        <div className="breed-name">{result.top_breed}</div>
        <div className="confidence">{result.top_breed_confidence.toFixed(1)}%</div>
        <div className="conf-label">Breed Confidence</div>
      </div>

      <h2 className="section-title">🏆 Top {top5.length} Predictions</h2>
      <ul className="top5-list">
        {top5.map((b, i) => (
          <li className="top5-item" key={b.breed}>
            <div className="top5-bar-wrap">
              <div
                className="top5-bar"
                style={{
                  width: bars ? `${Math.max(10, (b.confidence / max) * 100)}%` : "0%",
                  transitionDelay: `${i * 100}ms`,
                }}
              />
              <span className="breed-label">{b.breed}</span>
            </div>
            <span className="top5-pct">{b.confidence.toFixed(1)}%</span>
          </li>
        ))}
      </ul>

      {tier !== 3 && top5.length > 0 && (
        <TraitComparison
          breeds={top5.slice(0, 3).map((b) => b.breed)}
          defaultOpen={tier === 2}
        />
      )}

      <div className="info-footer">
        <span>🤖 {result.model_used}</span>
        <span>📄 {filename || result.filename || "unknown"}</span>
        <span>⏱️ {(result.total_time_ms || 0).toFixed(0)}ms</span>
      </div>

      {devMode && (
        <button className="btn outline-green" onClick={onExport} disabled={exporting}>
          {exporting ? "⏳ Generating ODT…" : "📄 Export to ODT"}
        </button>
      )}
    </div>
  );
}
