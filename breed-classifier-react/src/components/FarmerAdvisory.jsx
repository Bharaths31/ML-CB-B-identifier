import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import FarmerAdvisoryCard from "./FarmerAdvisoryCard.jsx";

/**
 * Advanced-drawer advisory. Gender and age are read-only model estimates; the
 * heavy rendering and all dynamic localization live in FarmerAdvisoryCard so
 * the drawer and the main workspace cannot diverge.
 */
export default function FarmerAdvisory({ result }) {
  const { t, lang } = useI18n();
  const initial = result?.knowledge;
  const [advisory, setAdvisory] = useState(initial);

  useEffect(() => {
    setAdvisory(initial);
  }, [initial, lang]);

  if (!initial || initial.error || initial.skipped) return null;

  const context = initial?.context || {};
  const gender = context.gender;
  const ageGroup = context.age_group;

  return (
    <section className="advanced-result-block farmer-advisory">
      <h4>{t("advisory.title")}</h4>
      <div className="farmer-context-row">
        <div className="farmer-context-group">
          <span className="farmer-context-label">{t("advisory.gender")}</span>
          <span className="badge badge-neutral">
            {gender ? t(`advisory.${String(gender).toLowerCase()}`) : t("hero.unknown")}
          </span>
        </div>
        <div className="farmer-context-group">
          <span className="farmer-context-label">{t("advisory.age")}</span>
          <span className="badge badge-neutral">
            {ageGroup ? t(`advisory.${String(ageGroup).toLowerCase()}`) : t("hero.unknown")}
          </span>
        </div>
      </div>
      <FarmerAdvisoryCard advisory={advisory} loading={false} latencyMs={null} />
    </section>
  );
}
