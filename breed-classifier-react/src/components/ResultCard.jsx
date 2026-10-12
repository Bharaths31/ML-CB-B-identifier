import { useState } from "react";

export default function ResultCard({ result, config, onReset }) {
  const [showDetails, setShowDetails] = useState(
    config?.display?.show_technical_details ?? false
  );

  if (!result) return null;

  const isOod = result.is_ood || false;
  const oodScore = result.ood_score;
  const breed = result.predicted_breed || result.breed || "Unknown";
  const species = result.predicted_species || result.species || "Cattle";
  const rawConfidence = result.confidence ?? 0;
  
  // Format percentage
  const confidencePct = Math.round(rawConfidence * 100);
  const topBreeds = result.top_k || result.top_breeds || [];
  const minBreeds = config?.display?.min_breeds ?? 1;
  const maxBreeds = config?.display?.max_breeds ?? (config?.display?.top_k ?? 3);
  const minConf = config?.display?.min_breed_confidence_pct ?? 0;

  // Filter candidates by min confidence, guaranteeing at least minBreeds
  let candidateList = topBreeds.filter((b) => {
    const val = typeof b.confidence === "number" ? b.confidence : (b.prob ? b.prob * 100 : 0);
    return val >= minConf;
  });
  if (candidateList.length < minBreeds) {
    candidateList = topBreeds.slice(0, minBreeds);
  }
  const displayedTop = candidateList.slice(0, maxBreeds);

  return (
    <div className="result-card-container">
      {/* OOD Alert Banner if sample is out of distribution */}
      {isOod && (
        <div className="alert-banner alert-warning" role="alert">
          <div className="alert-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
              <line x1="12" y1="9" x2="12" y2="13"></line>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
          </div>
          <div className="alert-content">
            <h4 className="alert-title">Unrecognized or Out-of-Distribution Sample</h4>
            <p className="alert-description">
              The AI detector flagged this image as significantly divergent from known training breeds.
              The prediction below has high uncertainty.
            </p>
            {oodScore !== undefined && (
              <span className="alert-meta">
                Energy Score: {typeof oodScore === "number" ? oodScore.toFixed(3) : oodScore}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Main Prediction Header */}
      <div className="prediction-main-card">
        <div className="prediction-header">
          <div className="species-badge-row">
            <span className={`species-pill species-${isOod ? "unknown" : species.toLowerCase()}`}>
              {isOod ? "OUT-OF-DISTRIBUTION" : species.toUpperCase()}
            </span>
            <span className="confidence-pill">
              {isOod ? "Uncertain Sample" : `${confidencePct}% Confidence`}
            </span>
          </div>

          <h2 className="predicted-breed-title">
            {isOod ? "Unrecognized Breed / Subject" : breed.replace(/_/g, " ")}
          </h2>

          <div className="confidence-meter-container">
            <div className="confidence-bar-bg">
              <div
                className={`confidence-bar-fill ${isOod ? "bar-warning" : "bar-success"}`}
                style={{ width: `${isOod ? 15 : Math.min(100, Math.max(0, confidencePct))}%` }}
              />
            </div>
            <div className="confidence-scale-labels">
              <span>{isOod ? "Below Threshold" : "0%"}</span>
              <span>{isOod ? "Outlier Detected" : "50%"}</span>
              <span>{isOod ? "Rejected" : "100%"}</span>
            </div>
          </div>
        </div>

        {/* Top Matches Breakdown */}
        {isOod ? (
          <div className="top-matches-section">
            <h3 className="section-subtitle">Breed Classification Withheld</h3>
            <div className="ood-advisory-box">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
              </svg>
              <span>
                To prevent false identifications, breed predictions are not shown for images that fail safety energy validation.
                Ensure the livestock subject is clearly centered and well-lit.
              </span>
            </div>
          </div>
        ) : (
          displayedTop.length > 0 && (
            <div className="top-matches-section">
              <h3 className="section-subtitle">Ranked Breed Probabilities</h3>
              <div className="breed-list">
                {displayedTop.map((item, idx) => {
                  const bName = (item.breed || item[0] || "").replace(/_/g, " ");
                  const bProb = typeof item.prob === "number" ? item.prob : (typeof item[1] === "number" ? item[1] : 0);
                  const bPct = Math.round(bProb * 100);
                  const isTop = idx === 0;

                  return (
                    <div key={idx} className={`breed-row ${isTop ? "breed-row-top" : ""}`}>
                      <div className="breed-row-header">
                        <span className="breed-rank">#{idx + 1}</span>
                        <span className="breed-name">{bName}</span>
                        <span className="breed-pct">{bPct}%</span>
                      </div>
                      <div className="breed-mini-bar-bg">
                        <div
                          className="breed-mini-bar-fill"
                          style={{ width: `${bPct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )
        )}

        {/* Technical Diagnostics Collapsible */}
        <div className="diagnostics-toggle-row">
          <button
            type="button"
            className="btn-text-action"
            onClick={() => setShowDetails(!showDetails)}
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              style={{ transform: showDetails ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }}
            >
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
            {showDetails ? "Hide Technical Details" : "Show Technical Details"}
          </button>
        </div>

        {showDetails && (
          <div className="diagnostics-panel">
            <div className="diagnostics-grid">
              <div className="diag-item">
                <span className="diag-label">Inference Time</span>
                <span className="diag-value">
                  {result.inference_time_ms ? `${result.inference_time_ms} ms` : "N/A"}
                </span>
              </div>
              <div className="diag-item">
                <span className="diag-label">Model Name</span>
                <span className="diag-value text-truncate" title={result.model_name}>
                  {result.model_name || config?.active_model || "Default"}
                </span>
              </div>
              <div className="diag-item">
                <span className="diag-label">OOD Evaluation</span>
                <span className="diag-value">
                  {config?.ood?.enabled ? (isOod ? "Flagged (OOD)" : "In-Distribution (Valid)") : "Disabled"}
                </span>
              </div>
              <div className="diag-item">
                <span className="diag-label">Species Soft Prob</span>
                <span className="diag-value">
                  {result.species_prob ? `${Math.round(result.species_prob * 100)}%` : "N/A"}
                </span>
              </div>
            </div>
          </div>
        )}

        <div className="card-footer-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onReset}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
              <path d="M21 3v5h-5"></path>
              <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
              <path d="M3 21v-5h5"></path>
            </svg>
            Classify Another Photo
          </button>
        </div>
      </div>
    </div>
  );
}
