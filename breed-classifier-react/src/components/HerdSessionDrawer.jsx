import { useI18n } from "../i18n/I18nProvider.jsx";
import { exportOdt, downloadBlob } from "../api.js";
import {
  clearSession,
  downloadTextFile,
  sessionStats,
  sessionToCsv,
  sessionToJson,
  sessionToOdtResults,
} from "../utils/sessionStore.js";

export default function HerdSessionDrawer({ open, entries, onClose, onCleared }) {
  const { t } = useI18n();
  if (!open) return null;

  const stats = sessionStats(entries);

  const handleClear = () => {
    if (!window.confirm(t("session.clearConfirm"))) return;
    clearSession();
    onCleared?.();
  };

  const handleOdt = async () => {
    if (!entries.length) return;
    try {
      const blob = await exportOdt(sessionToOdtResults(entries), "batch");
      downloadBlob(blob, `herd_session_${Date.now()}.odt`);
    } catch (err) {
      alert(err.message);
    }
  };

  return (
    <div className="session-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label={t("session.title")}>
      <aside className="session-drawer" onClick={(e) => e.stopPropagation()}>
        <header className="session-head">
          <div>
            <h3>{t("session.title")}</h3>
            <span>{t("session.count", { count: stats.total })}</span>
          </div>
          <button type="button" className="camera-close-icon" onClick={onClose} aria-label={t("actions.close")}>
            ✕
          </button>
        </header>

        <div className="session-stats-grid">
          <Stat label={t("session.total")} value={stats.total} />
          <Stat label={t("session.cattle")} value={stats.cattle} tone="green" />
          <Stat label={t("session.buffalo")} value={stats.buffalo} tone="accent" />
          <Stat label={t("session.flagged")} value={stats.flagged} tone={stats.flagged ? "red" : ""} />
          <Stat label={t("session.avgConfidence")} value={`${Math.round(stats.avgConfidence * 100)}%`} />
          {stats.avgQuality != null && (
            <Stat label={t("session.avgQuality")} value={`${Math.round(stats.avgQuality)}/100`} />
          )}
        </div>

        <div className="session-actions">
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={() => downloadTextFile(sessionToCsv(entries), `herd_session_${Date.now()}.csv`, "text/csv")}
            disabled={!entries.length}
          >
            {t("actions.exportCsv")}
          </button>
          <button
            type="button"
            className="btn btn-outline btn-sm"
            onClick={() =>
              downloadTextFile(sessionToJson(entries), `herd_session_${Date.now()}.json`, "application/json")
            }
            disabled={!entries.length}
          >
            {t("actions.exportJson")}
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={handleOdt} disabled={!entries.length}>
            {t("actions.exportOdt")}
          </button>
          <button type="button" className="btn btn-outline btn-sm session-clear" onClick={handleClear} disabled={!entries.length}>
            {t("actions.clear")}
          </button>
        </div>

        <div className="session-list">
          {entries.length === 0 ? (
            <p className="session-empty">{t("session.empty")}</p>
          ) : (
            entries.map((e) => (
              <article key={e.id} className={`session-row ${e.isOod ? "session-row-ood" : ""}`}>
                <div className="session-thumb">
                  {e.thumbnail ? <img src={e.thumbnail} alt="" /> : <span>🐄</span>}
                </div>
                <div className="session-row-main">
                  <div className="session-row-title">
                    <strong>{e.breed}</strong>
                    <span className="session-conf">{Math.round((e.confidence || 0) * 100)}%</span>
                  </div>
                  <div className="session-row-meta">
                    <span className={`session-species ${String(e.species).toLowerCase()}`}>{e.species}</span>
                    {e.isOod && <span className="session-ood">OOD</span>}
                    {e.qualityScore != null && <span>{e.qualityScore}/100</span>}
                    <span>{new Date(e.timestamp).toLocaleTimeString()}</span>
                  </div>
                </div>
              </article>
            ))
          )}
        </div>
      </aside>
    </div>
  );
}

function Stat({ label, value, tone = "" }) {
  return (
    <div className={`session-stat ${tone}`}>
      <span className="session-stat-value">{value}</span>
      <span className="session-stat-label">{label}</span>
    </div>
  );
}
