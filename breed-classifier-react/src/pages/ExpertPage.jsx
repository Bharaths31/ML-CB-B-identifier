import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getDevInfo, getEscalations, resolveEscalation } from "../api.js";
import { useSocket } from "../hooks/useSocket.js";
import { useI18n } from "../i18n/I18nProvider.jsx";
import ThemeToggle from "../components/ThemeToggle.jsx";

const FILTERS = ["pending", "verified", "rejected", "all"];

function metricCards(counts, t) {
  return [
    { key: "pending", label: t("expert.pending"), value: counts.pending_count || 0, tone: "amber" },
    { key: "verified", label: t("expert.verified"), value: counts.verified_count || 0, tone: "green" },
    { key: "rejected", label: t("expert.rejected"), value: counts.rejected_count || 0, tone: "red" },
  ];
}

function parseJson(value, fallback) {
  if (!value) return fallback;
  if (typeof value === "object") return value;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function breedOptions(devInfo, species) {
  if (!devInfo) return [];
  const isBuffalo = String(species || "").toLowerCase() === "buffalo";
  const key = isBuffalo ? "buffalo_classes" : "cattle_classes";
  const knownKey = isBuffalo ? "known_buffalo_classes" : "known_cattle_classes";
  const source = devInfo[knownKey] || devInfo[key] || {};
  return Object.keys(source).sort();
}

function softmaxBars(record) {
  const parsed = parseJson(record.top3, null) || parseJson(record.top3_json, []) || [];
  const top3 = Array.isArray(parsed) ? parsed : [];
  return top3.slice(0, 3).map((item) => ({
    breed: (item?.breed || item?.name || "").replace(/_/g, " "),
    prob:
      typeof item?.prob === "number"
        ? item.prob
        : typeof item?.confidence === "number"
        ? item.confidence / 100
        : 0,
  }));
}

function EscalationCard({ record, devInfo, onResolved, t }) {
  const [species, setSpecies] = useState(
    (record.expert_species || record.predicted_species || "Cattle").toLowerCase().includes("buff")
      ? "Buffalo"
      : "Cattle"
  );
  const initialBreed = record.expert_breed || record.predicted_breed || "";
  const [breed, setBreed] = useState(String(initialBreed).replace(/_/g, " "));
  const [notes, setNotes] = useState(record.expert_notes || "");
  const [promote, setPromote] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);
  const [xray, setXray] = useState(Boolean(record.attention_map_b64));

  const options = useMemo(() => breedOptions(devInfo, species), [devInfo, species]);
  const bars = useMemo(() => softmaxBars(record), [record]);
  const ood = parseJson(record.ood, null) || parseJson(record.ood_details, null) || {};
  const quality = parseJson(record.quality, null) || {};
  const resolved = record.status !== "pending";

  const submit = async (status) => {
    setMessage(null);
    let resolvedBreed = breed.trim();
    if (status === "verified") {
      if (!resolvedBreed) {
        setMessage(t("expert.breedRequired"));
        return;
      }
      const match = options.find(
        (option) => option.toLowerCase() === resolvedBreed.replace(/\s+/g, "_").toLowerCase()
          || option.toLowerCase() === resolvedBreed.toLowerCase()
      );
      if (options.length > 0 && !match) {
        setMessage(t("expert.breedUnknown"));
        return;
      }
      resolvedBreed = match || resolvedBreed;
    }
    setBusy(true);
    try {
      const response = await resolveEscalation(record.id, {
        status,
        expert_species: status === "verified" ? species.toLowerCase() : null,
        expert_breed: status === "verified" ? resolvedBreed : null,
        expert_notes: notes,
        promote_to_dataset: status === "verified" && promote,
      });
      const active = response?.escalation?.active_learning;
      if (active && active.promoted === false) {
        setMessage(active.error || t("expert.activeLearningFailed"));
      } else {
        setMessage(t("expert.saved"));
      }
      onResolved?.();
    } catch (err) {
      setMessage(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <article className={`expert-card expert-card-${record.status}`}>
      <div className="expert-card-media">
        {record.image_b64 ? (
          <div className="expert-image-stack">
            <img src={record.image_b64} alt={`Escalated ${record.predicted_breed}`} />
            {xray && record.attention_map_b64 && (
              <img src={record.attention_map_b64} alt="AI attention heatmap" className="expert-xray" />
            )}
          </div>
        ) : (
          <div className="expert-image-missing">No image payload</div>
        )}
        {record.attention_map_b64 && (
          <button type="button" className="btn btn-sm btn-outline" onClick={() => setXray((v) => !v)}>
            {xray ? "Hide heatmap" : "Show heatmap"}
          </button>
        )}
      </div>

      <div className="expert-card-body">
        <div className="expert-card-head">
          <div>
            <span className={`expert-level level-${record.confidence_level || 1}`}>
              Level {record.confidence_level || 1}
            </span>
            <h3>
              {record.predicted_species} · {(record.predicted_breed || "").replace(/_/g, " ")}
            </h3>
          </div>
          <span className={`expert-status status-${record.status}`}>{record.status}</span>
        </div>

        <div className="expert-meta-grid">
          <div>
            <span>{t("expert.gps")}</span>
            <strong>
              {record.gps_lat != null
                ? `${Number(record.gps_lat).toFixed(5)}, ${Number(record.gps_lng).toFixed(5)}`
                : record.location_label || "n/a"}
            </strong>
            <small>{new Date(record.created_at).toLocaleString()}</small>
          </div>
          <div>
            <span>AI confidence</span>
            <strong>{Math.round((Number(record.confidence) || 0) * 100)}%</strong>
            <small>{record.model_used || "unknown model"}</small>
          </div>
          <div>
            <span>FLW</span>
            <strong>{record.flw_id || "—"}</strong>
            <small>{record.filename}</small>
          </div>
        </div>

        <div className="expert-columns">
          <div>
            <h4>{t("expert.top3")}</h4>
            <div className="expert-bars">
              {bars.length === 0 && <span className="muted">—</span>}
              {bars.map((bar, index) => (
                <div key={`${bar.breed}-${index}`}>
                  <div className="expert-bar-label">
                    <span>{bar.breed}</span>
                    <span>{Math.round(bar.prob * 100)}%</span>
                  </div>
                  <div className="expert-bar-bg">
                    <div className="expert-bar-fill" style={{ width: `${Math.max(2, bar.prob * 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h4>{t("expert.ood")}</h4>
            <ul className="expert-telemetry">
              <li>OOD: {record.is_ood ? "flagged" : "in-distribution"}</li>
              {ood.energy != null && <li>Energy: {Number(ood.energy).toFixed(3)}</li>}
              {ood.msp != null && <li>MSP: {Number(ood.msp).toFixed(3)}</li>}
              {record.ood_score != null && <li>OOD score: {record.ood_score}</li>}
            </ul>
          </div>

          <div>
            <h4>{t("expert.quality")}</h4>
            <ul className="expert-telemetry">
              <li>Overall: {quality.overallScore ?? "—"}/100</li>
              <li>Sharpness: {quality.sharpnessScore ?? "—"}</li>
              <li>Exposure: {quality.exposureScore ?? "—"}</li>
              <li>Framing: {quality.framingScore ?? "—"}</li>
            </ul>
          </div>
        </div>

        {!resolved && (
          <div className="expert-form">
            <div className="expert-form-row">
              <label>
                {t("expert.species")}
                <select
                  className="form-select"
                  value={species}
                  onChange={(event) => {
                    setSpecies(event.target.value);
                    setBreed("");
                  }}
                >
                  <option>Cattle</option>
                  <option>Buffalo</option>
                </select>
              </label>
              <label>
                {t("expert.breed")}
                <input
                  className="form-input"
                  list={`breeds-${record.id}`}
                  value={breed}
                  placeholder={t("expert.searchBreed")}
                  onChange={(event) => setBreed(event.target.value)}
                />
                <datalist id={`breeds-${record.id}`}>
                  {options.map((option) => (
                    <option key={option} value={option} />
                  ))}
                </datalist>
              </label>
            </div>
            <label className="expert-notes">
              {t("expert.notes")}
              <textarea
                className="form-input"
                rows={2}
                value={notes}
                onChange={(event) => setNotes(event.target.value)}
                placeholder="Morphology notes, horn/hump/dewlap observations…"
              />
            </label>
            <label className="bpa-inline-check">
              <input type="checkbox" checked={promote} onChange={(event) => setPromote(event.target.checked)} />
              {t("expert.promote")}
            </label>
            <div className="cropper-actions">
              <button type="button" className="btn btn-danger" disabled={busy} onClick={() => submit("rejected")}>
                {t("expert.reject")}
              </button>
              <button type="button" className="btn btn-primary" disabled={busy || !breed} onClick={() => submit("verified")}>
                {busy ? "…" : t("expert.approve")}
              </button>
            </div>
          </div>
        )}

        {resolved && (
          <div className="expert-resolved">
            <strong>{record.status === "verified" ? "Verified" : "Rejected"}</strong>
            {record.expert_species && (
              <span>
                {" "}
                → {record.expert_species} / {(record.expert_breed || "").replace(/_/g, " ")}
              </span>
            )}
            {record.expert_notes && <p>{record.expert_notes}</p>}
            {record.reviewed_at && <small>Reviewed {new Date(record.reviewed_at).toLocaleString()}</small>}
          </div>
        )}

        {message && <div className="bpa-notice">{message}</div>}
      </div>
    </article>
  );
}

export default function ExpertPage() {
  const { t } = useI18n();
  const [filter, setFilter] = useState("pending");
  const [data, setData] = useState(null);
  const [devInfo, setDevInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { connected, escalationEvent } = useSocket();

  const load = useCallback(async (status = filter) => {
    setLoading(true);
    setError(null);
    try {
      const [queueResult, infoResult] = await Promise.allSettled([
        getEscalations(status),
        getDevInfo(),
      ]);
      if (queueResult.status === "fulfilled") {
        setData(queueResult.value);
      } else {
        throw queueResult.reason;
      }
      setDevInfo(infoResult.status === "fulfilled" ? infoResult.value : null);
      if (infoResult.status === "rejected") {
        setError(infoResult.reason?.message || "Breed class maps could not be loaded.");
      }
    } catch (err) {
      setError(err.message || "Could not load the expert queue");
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    load(filter);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter]);

  useEffect(() => {
    if (escalationEvent) load(filter);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [escalationEvent]);

  const counts = data || {};
  const escalations = data?.escalations || [];

  return (
    <div className="expert-page-container">
      <header className="site-header expert-header">
        <div className="header-brand">
          <div className="brand-icon-wrapper" aria-hidden="true">🩺</div>
          <div className="brand-info">
            <h1 className="brand-title">{t("expert.title")}</h1>
            <span className="brand-subtitle">{t("expert.subtitle")}</span>
          </div>
        </div>
        <div className="header-nav-right">
          <div className="status-badge">
            <span className={`status-dot ${connected ? "status-online" : "status-offline"}`} />
            <span className="status-text">{connected ? t("header.connected") : t("header.reconnecting")}</span>
          </div>
          <Link to="/" className="btn btn-outline">{t("expert.back")}</Link>
          <Link to="/dev" className="btn btn-outline">{t("header.dev")}</Link>
          <ThemeToggle />
        </div>
      </header>

      <main className="expert-main">
        <section className="expert-metrics">
          {metricCards(counts, t).map((metric) => (
            <div key={metric.key} className={`expert-metric metric-${metric.tone}`}>
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
            </div>
          ))}
        </section>

        <div className="expert-toolbar">
          <h2>{t("expert.queue")}</h2>
          <div className="expert-filters">
            {FILTERS.map((status) => (
              <button
                key={status}
                type="button"
                className={`preset-chip ${filter === status ? "active" : ""}`}
                onClick={() => setFilter(status)}
              >
                {t(`expert.filter.${status}`)}
              </button>
            ))}
            <button type="button" className="btn btn-sm btn-outline" onClick={() => load(filter)} disabled={loading}>
              Refresh
            </button>
          </div>
        </div>

        {error && <div className="error-banner">{error}</div>}
        {loading && <div className="expert-loading">{t("expert.loading")}</div>}
        {!loading && escalations.length === 0 && <div className="expert-empty">{t("expert.empty")}</div>}

        <section className="expert-grid">
          {escalations.map((record) => (
            <EscalationCard
              key={record.id}
              record={record}
              devInfo={devInfo}
              t={t}
              onResolved={() => load(filter)}
            />
          ))}
        </section>
      </main>
    </div>
  );
}
