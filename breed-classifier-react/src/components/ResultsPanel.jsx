import { useEffect, useState } from "react";

/** Single-image prediction result: hero card + top-5 bars + footer. */
export default function ResultsPanel({ result, devMode, filename, onExport, exporting }) {
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

  return (
    <div className="results fade-up">
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
