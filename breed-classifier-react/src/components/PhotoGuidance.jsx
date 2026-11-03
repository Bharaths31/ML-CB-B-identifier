import { useI18n } from "../i18n/I18nProvider.jsx";

/**
 * Photo-quality and OOD guidance for the farmer.
 *
 * The backend returns trilingual copy, so this component picks the current UI
 * language directly instead of maintaining a second translation table.
 */
export default function PhotoGuidance({ result }) {
  const { lang, t } = useI18n();

  const quality = result?.quality;
  const explanation = result?.ood_explanation;
  const isOod = Boolean(result?.is_ood ?? result?.ood);
  const qualityIssues = (quality?.issues || []).length > 0;
  const showOod = isOod && explanation && explanation.code !== "ok";

  if (!qualityIssues && !showOod) return null;

  const pick = (block) => (block && (block[lang] || block.en)) || "";
  const tips = showOod
    ? explanation.tips_by_language?.[lang] || explanation.tips_by_language?.en || []
    : quality.tips_by_language?.[lang] || quality.tips_by_language?.en || [];
  const severity = showOod ? explanation.severity : "warning";
  const title = showOod ? pick(explanation.title) : t("photoCheck.title");
  const message = showOod ? pick(explanation.message) : pick(quality.summary);

  return (
    <section className="advanced-result-block photo-guidance">
      <div className={`photo-guidance-banner photo-guidance-${severity}`}>
        <strong>{title}</strong>
        {!showOod && typeof quality.score === "number" && (
          <span className="photo-guidance-score">
            {t("photoCheck.score", { score: quality.score })}
          </span>
        )}
        {message && <p>{message}</p>}
      </div>

      {tips.length > 0 && (
        <>
          <h4 className="section-subtitle">{t("photoCheck.tips")}</h4>
          <ul className="farmer-list">
            {tips.map((tip, index) => (
              <li key={index}>{tip}</li>
            ))}
          </ul>
        </>
      )}

      {qualityIssues && (
        <p className="advanced-result-note">
          {t("photoCheck.issues")}: {quality.issues.join(", ")}
        </p>
      )}
    </section>
  );
}
