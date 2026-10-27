import { useI18n } from "../i18n/I18nProvider.jsx";

export const STATUS_META = {
  READY: { key: "quality.ready", icon: "✓", tone: "green" },
  TOO_BLURRY: { key: "quality.tooBlurry", icon: "◌", tone: "amber" },
  TOO_DARK: { key: "quality.tooDark", icon: "☾", tone: "amber" },
  OVEREXPOSED: { key: "quality.overexposed", icon: "☀", tone: "amber" },
  NO_SUBJECT_OR_TOO_FAR: { key: "quality.noSubject", icon: "⊕", tone: "amber" },
  PARTIALLY_OUT_OF_FRAME: { key: "quality.partial", icon: "⌗", tone: "amber" },
};

export function statusMeta(status) {
  return STATUS_META[status] || STATUS_META.READY;
}

function toneClass(status) {
  return status === "READY" ? "quality-good" : "quality-warn";
}

export function QualityPill({ quality, className = "" }) {
  const { t } = useI18n();
  if (!quality) return null;
  const meta = statusMeta(quality.status);
  return (
    <div
      className={`quality-pill ${toneClass(quality.status)} ${className}`}
      role="status"
      aria-live="polite"
    >
      <span className="quality-pill-icon" aria-hidden="true">{meta.icon}</span>
      <span className="quality-pill-text">{t(meta.key)}</span>
      <span className="quality-pill-score">Score: {Math.round(quality.overallScore)}%</span>
    </div>
  );
}

function MetricBar({ label, value, tone }) {
  return (
    <div className="quality-metric">
      <div className="quality-metric-head">
        <span>{label}</span>
        <span className="quality-metric-value">{Math.round(value)}</span>
      </div>
      <div className="quality-metric-track">
        <div
          className={`quality-metric-fill ${tone}`}
          style={{ width: `${Math.max(2, Math.min(100, value))}%` }}
        />
      </div>
    </div>
  );
}

export default function QualityBadge({ quality, compact = false }) {
  const { t } = useI18n();
  if (!quality) return null;

  const meta = statusMeta(quality.status);
  if (compact) {
    return (
      <span className={`quality-compact ${toneClass(quality.status)}`} title={t(meta.key)}>
        <span aria-hidden="true">{meta.icon}</span>
        {quality.overallScore}
      </span>
    );
  }

  return (
    <div className={`quality-panel ${toneClass(quality.status)}`}>
      <div className="quality-panel-head">
        <div>
          <span className="quality-panel-label">{t("quality.title")}</span>
          <div className="quality-status-line">
            <span className="quality-status-icon" aria-hidden="true">{meta.icon}</span>
            <span>{t(meta.key)}</span>
          </div>
        </div>
        <div className="quality-score-ring" aria-label={`${t("quality.score")} ${quality.overallScore}/100`}>
          <span className="quality-score-number">{quality.overallScore}</span>
          <span className="quality-score-max">/100</span>
        </div>
      </div>

      <div className="quality-metrics-grid">
        <MetricBar label={t("quality.sharpness")} value={quality.sharpnessScore} tone="tone-accent" />
        <MetricBar label={t("quality.lighting")} value={quality.exposureScore} tone="tone-amber" />
        <MetricBar label={t("quality.framing")} value={quality.framingScore} tone="tone-green" />
      </div>

      <p className="quality-hint">{t(`quality.hint.${quality.status === "READY" ? "ready" : statusToHint(quality.status)}`)}</p>
    </div>
  );
}

function statusToHint(status) {
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
