import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { fetchBreedKnowledgeOnline } from "../services/onDeviceInference.js";

function KeyValues({ data }) {
  if (!data || typeof data !== "object") return null;
  const entries = Object.entries(data).filter(([, value]) => (
    value !== null && value !== undefined && value !== ""
  ));
  if (!entries.length) return null;
  return (
    <div className="tips-kv">
      {entries.map(([key, value]) => (
        <div key={key} className="tips-kv-row">
          <span>{key.replace(/_/g, " ")}</span>
          <strong>{Array.isArray(value) ? value.join(" · ") : String(value)}</strong>
        </div>
      ))}
    </div>
  );
}

function List({ items }) {
  if (!Array.isArray(items) || !items.length) return null;
  return <ul className="tips-list">{items.map((item, index) => <li key={index}>{item}</li>)}</ul>;
}

export default function BreedTipsModal({ open, onClose, settings, result, gender, ageGroup }) {
  const { tBreed } = useI18n();
  const [state, setState] = useState({ loading: false, error: null, data: null });

  useEffect(() => {
    if (!open || !result) return undefined;
    let cancelled = false;
    setState({ loading: true, error: null, data: null });
    fetchBreedKnowledgeOnline({
      settings,
      species: result.species,
      breedSlug: result.breed_slug,
      gender,
      ageGroup,
    })
      .then((data) => {
        if (cancelled) return;
        if (!data.available) {
          setState({ loading: false, error: data.reason || "offline", data: null });
          return;
        }
        setState({ loading: false, error: null, data });
      })
      .catch((err) => {
        if (!cancelled) setState({ loading: false, error: err.message, data: null });
      });
    return () => { cancelled = true; };
  }, [open, result, settings, gender, ageGroup]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event) => { if (event.key === "Escape") onClose?.(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const advisory = state.data?.advisory || {};
  const knowledge = state.data?.knowledge || {};
  const breed = advisory.breed || knowledge.breed || {};
  const localNames = breed.local_names || {};

  const fieldTips = [
    "Photograph the full side profile in daylight, 2–3 m away, with the animal filling the frame.",
    ...(advisory.priority_actions || []),
  ];
  const genderKey = String(gender || "").toLowerCase();
  const ageKey = String(ageGroup || "").toLowerCase();
  if (genderKey.includes("female") && ageKey.includes("adult")) {
    fieldTips.push("Check the udder daily for heat, swelling or clots; use the CMT monthly for mastitis.");
  } else if (genderKey.includes("female") && (ageKey.includes("young") || ageKey.includes("calf"))) {
    fieldTips.push("Record heat cycles and target breeding weight; do not breed before the breed's recommended age.");
  } else if (genderKey.includes("male") && ageKey.includes("adult")) {
    fieldTips.push("Inspect the yoke, harness and hooves before work; provide shade and water during draught use.");
  } else if (ageKey.includes("calf")) {
    fieldTips.push("Colostrum within 1–2 hours, then 10% of body weight in milk daily; keep bedding dry.");
  }
  if (String(result?.species || "").toLowerCase().includes("buff")) {
    fieldTips.push("Buffaloes need shade and wallowing water in hot weather; bathe before milking for cleaner milk.");
  }
  fieldTips.push("Provide clean water and a mineral mixture daily, and record milk yield to spot sudden drops.");
  if (advisory.warning) fieldTips.push(advisory.warning);
  const uniqueTips = Array.from(new Set(fieldTips.filter(Boolean)));

  return (
    <div className="tips-overlay" role="presentation" onClick={onClose}>
      <section className="tips-modal" role="dialog" aria-modal="true" aria-label="Breed tips" onClick={(e) => e.stopPropagation()}>
        <header className="tips-head">
          <div>
            <span className="workspace-kicker">Breed Tips & Farmer Advisory</span>
            <h2>{tBreed(result?.breed_slug) || result?.breed || "Breed"}</h2>
            <p>{breed.utility_type || ""} {localNames.en && localNames.en !== breed.display_name ? `· ${localNames.en}` : ""}</p>
          </div>
          <button type="button" className="settings-close" onClick={onClose} aria-label="Close">✕</button>
        </header>

        {state.loading && <div className="tips-loading">Loading breed knowledge…</div>}

        {!state.loading && state.error && (
          <div className="tips-offline">
            <strong>📡 Offline Mode Active</strong>
            <p>
              Turn ON “Online Mode” in Settings and connect to the internet to load detailed
              Breed Tips & Farmer Advisory data.
            </p>
            <div className="tips-offline-tips">
              <strong>Quick tips while offline</strong>
              <List items={uniqueTips.slice(0, 4)} />
            </div>
            {state.error !== "offline_mode" && <small>{StateError(state.error)}</small>}
          </div>
        )}

        {!state.loading && !state.error && state.data && (
          <div className="tips-body">
            <section className="tips-section tips-highlight">
              <h3>💡 Practical tips for this animal</h3>
              <List items={uniqueTips} />
            </section>

            <section className="tips-section">
              <h3>🌍 Origin & Identity</h3>
              <KeyValues data={{
                origin: breed.origin_region,
                utility: breed.utility_type,
                local_names: localNames.en || undefined,
              }} />
            </section>

            <section className="tips-section">
              <h3>🥛 Milk, Yield & Economics</h3>
              <KeyValues data={advisory.production || knowledge.production} />
              <KeyValues data={advisory.economics || knowledge.economics} />
              {knowledge.estimated_market_price_inr && (
                <p className="tips-note">Market price: {knowledge.estimated_market_price_inr}</p>
              )}
            </section>

            <section className="tips-section">
              <h3>🌾 Fodder & Feeding</h3>
              <KeyValues data={advisory.nutrition || knowledge.feeding_and_care} />
            </section>

            <section className="tips-section">
              <h3>🛡️ Health & Disease Resistance</h3>
              <KeyValues data={advisory.health || knowledge.climate_and_disease_resistance} />
              <List items={advisory.health?.watch_for || knowledge.climate_and_disease_resistance?.watch_for} />
            </section>

            <section className="tips-section">
              <h3>🏛️ Government Schemes</h3>
              <List items={advisory.schemes || knowledge.govt_schemes} />
            </section>

            {advisory.warning && <p className="tips-note">{advisory.warning}</p>}
            {advisory.disclaimer && <p className="tips-note tips-disclaimer">{advisory.disclaimer}</p>}
          </div>
        )}

        <footer className="tips-footer">
          <span>{gender} · {ageGroup} · Online mode</span>
          <button type="button" className="btn btn-primary" onClick={onClose}>Close</button>
        </footer>
      </section>
    </div>
  );
}

function StateError(error) {
  if (error === "no_backend") return "Set a Backend URL in Settings.";
  if (error === "offline_mode") return "";
  return error;
}
