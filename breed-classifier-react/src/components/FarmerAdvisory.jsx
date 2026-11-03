import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { recalculateAdvisory } from "../api.js";
import FarmerAdvisoryCard from "./FarmerAdvisoryCard.jsx";

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
 * Advanced-drawer advisory with gender/age overrides. The heavy rendering and
 * all dynamic localization live in FarmerAdvisoryCard so the drawer and the
 * main workspace cannot diverge.
 */
export default function FarmerAdvisory({ result }) {
  const { t, lang } = useI18n();
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

  useEffect(() => {
    setAdvisory(initial);
    setGender(initial?.context?.gender || null);
    setAgeGroup(initial?.context?.age_group || null);
    setLatency(null);
    setError("");
  }, [initial, lang]);

  if (!initial || initial.error || initial.skipped) return null;

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
      setError(err?.message || t("result.recalculating"));
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

  return (
    <section className="advanced-result-block farmer-advisory">
      <h4>{t("advisory.title")}</h4>
      <div className="farmer-context-row">
        <div className="farmer-context-group" role="group" aria-label={t("advisory.gender")}>
          <span className="farmer-context-label">{t("advisory.gender")}</span>
          {GENDERS.map((option) => (
            <button
              key={option.key}
              type="button"
              className={`hero-pill ${gender === option.key ? "hero-pill-active" : ""}`}
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
              className={`hero-pill ${ageGroup === option.key ? "hero-pill-active" : ""}`}
              onClick={() => chooseAge(option.key)}
              disabled={busy}
              aria-pressed={ageGroup === option.key}
            >
              {t(option.labelKey)}
            </button>
          ))}
        </div>
      </div>
      {error && <p className="advanced-result-note">{error}</p>}
      <FarmerAdvisoryCard advisory={advisory} loading={busy} latencyMs={latency} />
    </section>
  );
}
