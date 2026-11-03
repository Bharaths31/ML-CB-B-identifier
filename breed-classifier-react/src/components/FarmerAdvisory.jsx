import { useEffect, useMemo, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { recalculateAdvisory } from "../api.js";

const GENDERS = [
  { key: "Female", labelKey: "advisory.female" },
  { key: "Male", labelKey: "advisory.male" },
];
const AGES = [
  { key: "Calf", labelKey: "advisory.calf" },
  { key: "Young", labelKey: "advisory.young" },
  { key: "Adult", labelKey: "advisory.adult" },
];

/**
 * Farmer / general-public advisory card.
 *
 * The backend attaches `knowledge` to every in-distribution prediction. The
 * gender/age chips re-query POST /api/advisory-recalculate (~0.1 ms) so the
 * card updates instantly without re-running image inference.
 */
export default function FarmerAdvisory({ result }) {
  const { t } = useI18n();
  const initial = result?.knowledge;

  const [advisory, setAdvisory] = useState(initial);
  const [gender, setGender] = useState(initial?.context?.gender || null);
  const [ageGroup, setAgeGroup] = useState(initial?.context?.age_group || null);
  const [latency, setLatency] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const breed = advisory?.breed;
  const species = breed?.species || result?.predicted_species || result?.species;
  const slug = breed?.slug;

  // A new prediction must reset the local toggles.
  useEffect(() => {
    setAdvisory(initial);
    setGender(initial?.context?.gender || null);
    setAgeGroup(initial?.context?.age_group || null);
    setLatency(null);
    setError("");
  }, [initial]);

  const usable = useMemo(
    () => Boolean(advisory && !advisory.error && !advisory.skipped && advisory.headline),
    [advisory]
  );

  if (!usable) return null;

  const update = async (nextGender, nextAge) => {
    if (!slug || !species) return;
    setBusy(true);
    setError("");
    try {
      const data = await recalculateAdvisory({
        species,
        breed: slug,
        gender: nextGender,
        ageGroup: nextAge,
      });
      setAdvisory(data.advisory);
      setLatency(data.latency_ms);
    } catch (err) {
      setError(err?.message || "Could not refresh advisory");
    } finally {
      setBusy(false);
    }
  };

  const chooseGender = (value) => {
    const next = gender === value ? null : value;
    setGender(next);
    update(next, ageGroup);
  };
  const chooseAge = (value) => {
    const next = ageGroup === value ? null : value;
    setAgeGroup(next);
    update(gender, next);
  };

  const milk = advisory.production?.milk;
  const draught = advisory.production?.draught;
  const nutrition = advisory.nutrition || {};
  const health = advisory.health || {};
  const economics = advisory.economics || {};

  return (
    <section className="advanced-result-block farmer-advisory">
      <h4>{t("advisory.title")}</h4>
      <p className="advanced-result-note">
        {breed.display_name}
        {breed.origin_region ? ` · ${breed.origin_region}` : ""}
        {breed.utility_type ? ` · ${breed.utility_type}` : ""}
      </p>

      <div className="farmer-context-row">
        <div className="farmer-context-group" role="group" aria-label={t("advisory.gender")}>
          <span className="farmer-context-label">{t("advisory.gender")}</span>
          {GENDERS.map((option) => (
            <button
              key={option.key}
              type="button"
              className={`btn btn-sm ${gender === option.key ? "btn-primary" : "btn-outline"}`}
              onClick={() => chooseGender(option.key)}
              disabled={busy}
              aria-pressed={gender === option.key}
            >
              {t(option.labelKey)}
            </button>
          ))}
        </div>
        <div className="farmer-context-group" role="group" aria-label={t("advisory.age")}>
          <span className="farmer-context-label">{t("advisory.age")}</span>
          {AGES.map((option) => (
            <button
              key={option.key}
              type="button"
              className={`btn btn-sm ${ageGroup === option.key ? "btn-primary" : "btn-outline"}`}
              onClick={() => chooseAge(option.key)}
              disabled={busy}
              aria-pressed={ageGroup === option.key}
            >
              {t(option.labelKey)}
            </button>
          ))}
        </div>
      </div>

      {latency != null && (
        <p className="advanced-result-note">
          {t("advisory.updatedIn", { ms: latency.toFixed(2) })}
        </p>
      )}
      {error && <p className="advanced-result-note">{error}</p>}

      <p className="farmer-headline">{advisory.headline}</p>

      {Array.isArray(advisory.priority_actions) && advisory.priority_actions.length > 0 && (
        <>
          <h4 className="section-subtitle">{t("advisory.actions")}</h4>
          <ol className="farmer-actions">
            {advisory.priority_actions.map((action, index) => (
              <li key={index}>{action}</li>
            ))}
          </ol>
        </>
      )}

      <h4 className="section-subtitle">{t("advisory.nutrition")}</h4>
      <div className="advanced-quality-grid">
        <div><span>{t("advisory.greenFodder")}</span><strong>{nutrition.green_fodder || "—"}</strong></div>
        <div><span>{t("advisory.dryFodder")}</span><strong>{nutrition.dry_fodder || "—"}</strong></div>
        <div><span>{t("advisory.concentrate")}</span><strong>{nutrition.concentrate || "—"}</strong></div>
        <div><span>{t("advisory.minerals")}</span><strong>{nutrition.mineral_mixture || "—"}</strong></div>
        <div><span>{t("advisory.water")}</span><strong>{nutrition.water || "—"}</strong></div>
      </div>
      {nutrition.notes && <p className="advanced-result-note">{nutrition.notes}</p>}

      <h4 className="section-subtitle">{t("advisory.production")}</h4>
      {milk ? (
        <div className="advanced-quality-grid">
          <div><span>{t("advisory.dailyYield")}</span><strong>{milk.daily_yield}</strong></div>
          <div><span>{t("advisory.lactation")}</span><strong>{milk.lactation_yield}</strong></div>
          <div><span>{t("advisory.fat")}</span><strong>{milk.fat}</strong></div>
          <div><span>{t("advisory.income")}</span><strong>{milk.estimated_monthly_income_inr}</strong></div>
        </div>
      ) : draught ? (
        <div className="advanced-quality-grid">
          <div><span>{t("advisory.work")}</span><strong>{draught.work_capacity}</strong></div>
          <div><span>{t("advisory.load")}</span><strong>{draught.load}</strong></div>
        </div>
      ) : (
        <p className="advanced-result-note">{t("advisory.noProduction")}</p>
      )}
      {draught?.note && <p className="advanced-result-note">{draught.note}</p>}

      <h4 className="section-subtitle">{t("advisory.health")}</h4>
      {Array.isArray(health.vaccination) && (
        <ul className="farmer-list">
          {health.vaccination.map((item, index) => <li key={index}>{item}</li>)}
        </ul>
      )}
      {Array.isArray(health.watch_for) && (
        <>
          <span className="farmer-context-label">{t("advisory.watchFor")}</span>
          <ul className="farmer-list">
            {health.watch_for.map((item, index) => <li key={index}>{item}</li>)}
          </ul>
        </>
      )}
      {health.deworming_schedule && (
        <p className="advanced-result-note">
          {t("advisory.deworming")}: {health.deworming_schedule.join(" → ")}
        </p>
      )}

      <h4 className="section-subtitle">{t("advisory.verify")}</h4>
      <ul className="farmer-list">
        {(advisory.physical_markers || []).map((marker, index) => (
          <li key={index}>{marker}</li>
        ))}
      </ul>

      <div className="advanced-quality-grid">
        <div>
          <span>{t("advisory.price")}</span>
          <strong>{economics.market_price || "—"}</strong>
        </div>
        {advisory.climate?.heat_tolerance && (
          <div>
            <span>{t("advisory.heat")}</span>
            <strong>{advisory.climate.heat_tolerance}</strong>
          </div>
        )}
      </div>

      {Array.isArray(advisory.schemes) && advisory.schemes.length > 0 && (
        <>
          <h4 className="section-subtitle">{t("advisory.schemes")}</h4>
          <ul className="farmer-list">
            {advisory.schemes.map((scheme, index) => <li key={index}>{scheme}</li>)}
          </ul>
        </>
      )}

      {Array.isArray(advisory.warnings) && advisory.warnings.length > 0 && (
        <ul className="farmer-list farmer-warnings">
          {advisory.warnings.map((warning, index) => <li key={index}>{warning}</li>)}
        </ul>
      )}

      {breed.data_confidence === "regional_default" && (
        <p className="advanced-result-note">{t("advisory.regionalDefault")}</p>
      )}
      <p className="farmer-disclaimer">{advisory.disclaimer}</p>
    </section>
  );
}
