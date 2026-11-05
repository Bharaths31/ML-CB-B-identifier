import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import {
  DEFAULT_ADVISORY_API_URL,
  fetchAdvisoryData,
} from "../services/onlineAdvisory.js";

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
  return (
    <ul className="tips-list">
      {items.map((item, index) => (
        <li key={index}>
          {typeof item === "string" ? item : (
            <>
              <strong>{item.name}</strong>
              {item.description ? ` — ${item.description}` : ""}
            </>
          )}
        </li>
      ))}
    </ul>
  );
}

export default function BreedTipsModal({ open, onClose, settings, result, gender, ageGroup }) {
  const { tBreed } = useI18n();
  const [state, setState] = useState({ loading: false, error: null, data: null });

  useEffect(() => {
    if (!open || !result) return undefined;
    let cancelled = false;
    setState({ loading: true, error: null, data: null });
    const apiUrl = settings?.advisoryApiUrl || DEFAULT_ADVISORY_API_URL;
    fetchAdvisoryData(
      result.species,
      result.breed_slug,
      gender,
      ageGroup,
      settings?.language || "en",
      apiUrl,
    )
      .then((payload) => {
        if (cancelled) return;
        if (!payload?.ok) {
          setState({ loading: false, error: payload, data: null });
          return;
        }
        setState({ loading: false, error: null, data: payload });
      })
      .catch((err) => {
        if (!cancelled) {
          setState({
            loading: false,
            error: { message: err.message, error: "advisory_network_error" },
            data: null,
          });
        }
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

  const breed = state.data?.breed || {};
  const advisory = state.data?.advisory || {};
  const cards = Array.isArray(advisory.cards) ? advisory.cards : [];
  const schemes = advisory.schemes?.items || breed.schemes || [];
  const vaccination = advisory.vaccination_schedule;
  const context = advisory.context || {};
  const marketPrice = breed.market_price || "";
  const milk = breed.milk_yield;
  const feeding = breed.feeding || {};
  const climate = breed.climate || {};
  const localNames = breed.local_names || {};

  const fallbackTips = [
    "Photograph the full side profile in daylight, 2–3 m away, with the animal filling the frame.",
    "Provide clean water and a mineral mixture daily, and record milk yield to spot sudden drops.",
    "Keep vaccination and deworming dates in a notebook or phone reminder.",
  ];

  return (
    <div className="tips-overlay" role="presentation" onClick={onClose}>
      <section className="tips-modal" role="dialog" aria-modal="true" aria-label="Breed tips" onClick={(e) => e.stopPropagation()}>
        <header className="tips-head">
          <div>
            <span className="workspace-kicker">Breed Tips & Farmer Advisory</span>
            <h2>{breed.name || tBreed(result?.breed_slug) || result?.breed || "Breed"}</h2>
            <p>{breed.utility_type || ""}</p>
          </div>
          <button type="button" className="settings-close" onClick={onClose} aria-label="Close">✕</button>
        </header>

        {state.loading && <div className="tips-loading">Loading breed knowledge…</div>}

        {!state.loading && state.error && (
          <div className="tips-offline">
            <strong>📡 {state.error.offline ? "Offline Mode Active" : "Breed tips unavailable"}</strong>
            <p>{state.error.message || "The on-device result is still valid. Check your connection and try again."}</p>
            <div className="tips-offline-tips">
              <strong>Quick tips while offline</strong>
              <List items={fallbackTips} />
            </div>
            {state.error.error && <small>{state.error.error}</small>}
          </div>
        )}

        {!state.loading && !state.error && state.data && (
          <div className="tips-body">
            <section className="tips-section tips-highlight">
              <h3>💡 {advisory.title}</h3>
              <p>{advisory.summary}</p>
            </section>

            {cards.map((card) => (
              <section className="tips-section" key={card.id}>
                <h3>{card.title}</h3>
                <p>{card.body}</p>
              </section>
            ))}

            <section className="tips-section">
              <h3>🌍 Origin & Identity</h3>
              <KeyValues data={{
                origin: breed.origin,
                utility: breed.utility_type,
                local_names: localNames.en || undefined,
              }} />
            </section>

            <section className="tips-section">
              <h3>🥛 Milk, Yield & Economics</h3>
              {context.milk_yield_visible && milk ? (
                <KeyValues data={{
                  daily_liters: milk.daily_liters,
                  lactation_yield: milk.lactation_yield,
                  fat_percentage: milk.fat_percentage,
                }} />
              ) : (
                <p className="tips-note">Milk yield is hidden for this gender/age group.</p>
              )}
              {marketPrice && <p className="tips-note">Market price: {marketPrice}</p>}
            </section>

            <section className="tips-section">
              <h3>🌾 Fodder & Feeding</h3>
              <KeyValues data={feeding} />
            </section>

            <section className="tips-section">
              <h3>🛡️ Health & Disease Resistance</h3>
              <KeyValues data={climate} />
            </section>

            {vaccination?.items?.length > 0 && (
              <section className="tips-section">
                <h3>💉 {vaccination.title}</h3>
                <div className="tips-kv">
                  {vaccination.items.map((item, index) => (
                    <div className="tips-kv-row" key={index}>
                      <span>{item.age}</span>
                      <strong>{item.item}{item.note ? ` — ${item.note}` : ""}</strong>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <section className="tips-section">
              <h3>🏛️ {advisory.schemes?.title || "Government Schemes"}</h3>
              <p className="tips-note">{advisory.schemes?.note}</p>
              <List items={schemes} />
            </section>

            {advisory.disclaimer && <p className="tips-note tips-disclaimer">{advisory.disclaimer}</p>}
          </div>
        )}

        <footer className="tips-footer">
          <span>{gender} · {ageGroup} · Live advisory</span>
          <button type="button" className="btn btn-primary" onClick={onClose}>Close</button>
        </footer>
      </section>
    </div>
  );
}
