import { useI18n } from "../i18n/I18nProvider.jsx";

function Field({ label, value }) {
  if (!value) return null;
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

/** Public-facing breed heritage panel, fed by /api/breed-knowledge. */
export default function BreedEncyclopedia({ entry, summary, loading }) {
  const { t, lang } = useI18n();

  if (loading) return <p className="advanced-result-note">{t("encyclopedia.loading")}</p>;
  if (!entry) return <p className="advanced-result-note">{t("encyclopedia.unavailable")}</p>;

  const names = entry.local_names || {};
  const localPairs = Object.entries(names).filter(([code, name]) => code !== "en" && name && name !== names.en);
  const milk = entry.milk_yield || {};
  const climate = entry.climate_and_disease_resistance || {};
  const isRare = entry.data_confidence === "regional_default"
    || String(entry.utility_type || "").toLowerCase().includes("ornamental");

  return (
    <div className="breed-encyclopedia">
      <header className="encyclopedia-head">
        <div>
          <h3>{entry.display_name}</h3>
          <p>{entry.origin_region}</p>
        </div>
        <span className={`encyclopedia-badge ${isRare ? "is-rare" : ""}`}>
          {isRare ? "NBAGR · " + t("encyclopedia.conservation") : entry.utility_type}
        </span>
      </header>

      <div className="advanced-quality-grid">
        <Field label={t("encyclopedia.localNames")}
               value={localPairs.map(([, name]) => name).join(" · ") || entry.display_name} />
        <Field label={t("encyclopedia.nativeTract")} value={entry.origin_region} />
        <Field label={t("encyclopedia.utility")} value={entry.utility_type} />
        <Field label={t("encyclopedia.market")} value={entry.estimated_market_price_inr} />
      </div>

      <h4 className="section-subtitle">{t("encyclopedia.milk")}</h4>
      <div className="advanced-quality-grid">
        <Field label={t("advisory.dailyYield")} value={milk.daily_liters} />
        <Field label={t("advisory.lactation")} value={milk.lactation_yield} />
        <Field label={t("advisory.fat")} value={milk.fat_percentage} />
        <Field label="β-casein" value={milk.beta_casein} />
      </div>

      <h4 className="section-subtitle">{t("encyclopedia.climate")}</h4>
      <div className="advanced-quality-grid">
        <Field label={t("advisory.heat")} value={climate.heat_tolerance} />
        <Field label="Ticks / parasites" value={climate.tick_parasite_resistance} />
        <Field label={t("advisory.water")} value={entry.feeding_and_care?.water_liters_per_day} />
      </div>
      {climate.disease_notes && <p className="advanced-result-note">{climate.disease_notes}</p>}

      {Array.isArray(entry.physical_markers) && (
        <>
          <h4 className="section-subtitle">{t("advisory.verify")}</h4>
          <ul className="farmer-list">
            {entry.physical_markers.map((marker, index) => <li key={index}>{marker}</li>)}
          </ul>
        </>
      )}

      {Array.isArray(entry.govt_schemes) && entry.govt_schemes.length > 0 && (
        <>
          <h4 className="section-subtitle">{t("encyclopedia.schemes")}</h4>
          <ul className="farmer-list">
            {entry.govt_schemes.map((scheme, index) => <li key={index}>{scheme}</li>)}
          </ul>
        </>
      )}

      <p className="encyclopedia-note">
        {isRare ? t("encyclopedia.rare") : t("encyclopedia.notified")}
      </p>
      {entry.note && <p className="advanced-result-note">{entry.note}</p>}
      {summary?.origin_region && !entry.origin_region && (
        <p className="advanced-result-note">{summary.origin_region}</p>
      )}
    </div>
  );
}
