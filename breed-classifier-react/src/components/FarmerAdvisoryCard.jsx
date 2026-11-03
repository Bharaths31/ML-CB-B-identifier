import { useI18n } from "../i18n/I18nProvider.jsx";

function Row({ label, value }) {
  if (!value) return null;
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

/** Pure presentation of a `get_farmer_advisory` payload. */
export default function FarmerAdvisoryCard({ advisory, loading, latencyMs }) {
  const { t, renderAdvisory } = useI18n();
  const localized = renderAdvisory(advisory);

  if (loading && !advisory) {
    return <p className="advanced-result-note">{t("result.recalculating")}</p>;
  }
  if (!localized || localized.error || localized.skipped) {
    return <p className="advanced-result-note">{t("encyclopedia.unavailable")}</p>;
  }

  const milk = localized.production?.milk;
  const draught = localized.production?.draught;
  const nutrition = localized.nutrition || {};
  const health = localized.health || {};
  const economics = localized.economics || {};
  const breed = localized.breed || {};

  return (
    <div className={`farmer-advisory-card ${loading ? "is-loading" : ""}`}>
      <div className="advisory-headline-row">
        <p className="farmer-headline">{localized.headline}</p>
        {latencyMs != null && (
          <span className="advisory-latency">
            {t("result.recalculatedIn", { ms: Number(latencyMs).toFixed(2) })}
          </span>
        )}
      </div>

      <div className="advisory-milk-row">
        <article className="advisory-milk-card">
          <span className="advisory-card-kicker">{t("advisory.dailyYield")}</span>
          <strong>{milk?.daily_yield || "—"}</strong>
          <span className="advisory-milk-meta">
            {milk?.fat ? `${milk.fat} · ${t("advisory.fat")}` : ""}
            {milk?.beta_casein ? ` · ${milk.beta_casein}` : ""}
          </span>
        </article>
        <article className="advisory-milk-card">
          <span className="advisory-card-kicker">{t("advisory.lactation")}</span>
          <strong>{milk?.lactation_yield || "—"}</strong>
          <span className="advisory-milk-meta">{milk?.lactation_length || ""}</span>
        </article>
        <article className="advisory-milk-card advisory-income-card">
          <span className="advisory-card-kicker">{t("advisory.income")}</span>
          <strong>{milk?.estimated_monthly_income_inr || economics.market_price || "—"}</strong>
          <span className="advisory-milk-meta">{milk?.income_basis || ""}</span>
        </article>
      </div>

      {!milk && draught && (
        <div className="advanced-quality-grid">
          <Row label={t("advisory.work")} value={draught.work_capacity} />
          <Row label={t("advisory.load")} value={draught.load} />
        </div>
      )}
      {draught?.note && <p className="advanced-result-note">{draught.note}</p>}

      {Array.isArray(localized.priority_actions) && (
        <>
          <h4 className="section-subtitle">{t("advisory.actions")}</h4>
          <ol className="farmer-actions">
            {localized.priority_actions.map((action, index) => (
              <li key={index}>{action}</li>
            ))}
          </ol>
        </>
      )}

      <h4 className="section-subtitle">{t("advisory.nutrition")}</h4>
      <div className="advanced-quality-grid">
        <Row label={t("advisory.greenFodder")} value={nutrition.green_fodder} />
        <Row label={t("advisory.dryFodder")} value={nutrition.dry_fodder} />
        <Row label={t("advisory.concentrate")} value={nutrition.concentrate} />
        <Row label={t("advisory.minerals")} value={nutrition.mineral_mixture} />
        <Row label={t("advisory.water")} value={nutrition.water} />
      </div>
      {nutrition.notes && <p className="advanced-result-note">{nutrition.notes}</p>}

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

      <div className="advisory-economics-row">
        <div className="advisory-price">
          <span className="advisory-card-kicker">{t("advisory.price")}</span>
          <strong>{economics.market_price || breed.origin_region || "—"}</strong>
        </div>
        {Array.isArray(localized.schemes) && (
          <div className="advisory-schemes">
            <span className="advisory-card-kicker">{t("localized.schemes")}</span>
            <ul>
              {localized.schemes.slice(0, 4).map((scheme, index) => (
                <li key={index}>{scheme}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {Array.isArray(localized.warnings) && localized.warnings.length > 0 && (
        <ul className="farmer-list farmer-warnings">
          {localized.warnings.map((warning, index) => <li key={index}>{warning}</li>)}
        </ul>
      )}
      {breed.data_confidence === "regional_default" && (
        <p className="advanced-result-note">{t("advisory.regionalDefault")}</p>
      )}
      <p className="farmer-disclaimer">{localized.disclaimer}</p>
    </div>
  );
}

