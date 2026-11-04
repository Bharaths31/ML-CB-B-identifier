import { useI18n } from "../i18n/I18nProvider.jsx";

function pct(value) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, Math.min(100, n)) : 0;
}

/** Confidence meter used by every hero card. */
function Meter({ value, tone = "green" }) {
  const width = pct(value);
  return (
    <div className="hero-meter" aria-hidden="true">
      <div className={`hero-meter-fill tone-${tone}`} style={{ width: `${width}%` }} />
    </div>
  );
}

/**
 * Reviewer-mandated "Big 4" grid: Species, Breed, Gender, Age Group.
 * Gender and age are read-only model estimates (no manual override).
 */
export default function HeroFourGrid({
  result,
  breedKnowledge,
  showSpeciesConfidence = true,
}) {
  const { t, lang, tBreed, tDynamic } = useI18n();
  const isOod = Boolean(result?.is_ood ?? result?.ood);
  const species = result?.predicted_species || result?.species || "";
  const speciesKey = String(species).toLowerCase().startsWith("buff") ? "buffalo" : "cattle";
  const speciesIcon = speciesKey === "buffalo" ? "🐃" : "🐄";
  const speciesLabel = isOod
    ? t("hero.unknown")
    : tDynamic("species", species) || species;
  const breedRaw = result?.predicted_breed || result?.top_breed || "";
  const breed = isOod ? t("hero.unknown") : tBreed(breedRaw);
  const top = Array.isArray(result?.top5_breeds) ? result.top5_breeds : [];
  const alternatives = top.slice(1, 4).filter((item) => item?.breed);
  const localNames = breedKnowledge?.local_names || result?.knowledge?.breed?.local_names || {};
  const localName = localNames[lang] && localNames[lang] !== localNames.en ? localNames[lang] : null;
  const origin = breedKnowledge?.origin_region || result?.knowledge?.breed?.origin_region || "";
  const genderBlock = result?.gender || {};
  const ageBlock = result?.age_group || {};

  return (
    <div className="hero-four-grid" aria-label={t("hero.species")}>
      {/* 1 — Species */}
      <article className={`hero-card hero-species hero-species-${speciesKey} ${isOod ? "hero-muted" : ""}`}>
        <header className="hero-card-head">
          <span className="hero-card-label">{t("hero.species")}</span>
          <span className="hero-species-icon" aria-hidden="true">{speciesIcon}</span>
        </header>
        <h3 className="hero-value">{speciesLabel}</h3>
        {showSpeciesConfidence && (
          <>
            <div className="hero-confidence-row">
              <span>{Math.round(pct(result?.species_confidence))}% {t("hero.confidence")}</span>
            </div>
            <Meter value={result?.species_confidence} tone={speciesKey === "buffalo" ? "teal" : "amber"} />
          </>
        )}
      </article>

      {/* 2 — Breed */}
      <article className={`hero-card hero-breed ${isOod ? "hero-muted" : ""}`}>
        <header className="hero-card-head">
          <span className="hero-card-label">{t("hero.breed")}</span>
          <span className="hero-confidence-chip">
            {Math.round(pct(result?.top_breed_confidence))}%
          </span>
        </header>
        <h3 className="hero-value">{isOod ? t("hero.unknown") : breed || t("hero.unknown")}</h3>
        {localName && <p className="hero-local-name">{localName}</p>}
        {origin && <p className="hero-origin">{origin}</p>}
        {alternatives.length > 0 && !isOod && (
          <div className="hero-alt-pills" aria-label={t("hero.topAlternatives")}>
            {alternatives.map((item) => (
              <span key={item.breed} className="hero-alt-pill">
                {tBreed(item.breed)} · {Math.round(pct(item.confidence))}%
              </span>
            ))}
          </div>
        )}
        <Meter value={result?.top_breed_confidence} tone="green" />
      </article>

      {/* 3 — Gender */}
      <article className={`hero-card hero-gender ${isOod ? "hero-muted" : ""}`}>
        <header className="hero-card-head">
          <span className="hero-card-label">{t("hero.gender")}</span>
          <span className="hero-confidence-chip">
            {Math.round(pct(genderBlock.confidence))}%
          </span>
        </header>
        <h3 className="hero-value">
          {isOod ? t("hero.unknown") : (tDynamic("gender", genderBlock.key) || t("hero.unknown"))}
        </h3>
        {genderBlock.method && (
          <p className="hero-method">
            {genderBlock.method === "clip_zeroshot"
              ? t("hero.method.clip")
              : genderBlock.method === "morphological_heuristic"
              ? t("hero.method.heuristic")
              : t("hero.genderNote")}
          </p>
        )}
        <p className="hero-hint">{t("hero.genderNote")}</p>
      </article>

      {/* 4 — Age group */}
      <article className={`hero-card hero-age ${isOod ? "hero-muted" : ""}`}>
        <header className="hero-card-head">
          <span className="hero-card-label">{t("hero.age")}</span>
          <span className="hero-confidence-chip">
            {Math.round(pct(ageBlock.confidence))}%
          </span>
        </header>
        <h3 className="hero-value">
          {isOod ? t("hero.unknown") : (tDynamic("age", ageBlock.key) || t("hero.unknown"))}
        </h3>
        <p className="hero-hint" title={t("hero.ageTooltip")}>
          ℹ️ {t("hero.ageTooltip")}
        </p>
      </article>
    </div>
  );
}
