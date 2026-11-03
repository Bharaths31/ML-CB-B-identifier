import { useEffect } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";

const TIPS = [
  { key: "1", icon: "🐄", visual: "side-profile", titleKey: "photoGuide.tip1Title", bodyKey: "photoGuide.tip1Body" },
  { key: "2", icon: "☀️", visual: "daylight", titleKey: "photoGuide.tip2Title", bodyKey: "photoGuide.tip2Body" },
  { key: "3", icon: "🎯", visual: "single", titleKey: "photoGuide.tip3Title", bodyKey: "photoGuide.tip3Body" },
];

/** Modal "how to snap" guide with three visual mini-cards. */
export default function PhotoGuideModal({ open, onClose }) {
  const { t } = useI18n();

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="photo-guide-overlay" onClick={onClose} role="dialog" aria-modal="true"
         aria-label={t("photoGuide.title")}>
      <section className="photo-guide-modal" onClick={(event) => event.stopPropagation()}>
        <header className="photo-guide-head">
          <div>
            <h3>{t("photoGuide.title")}</h3>
            <p>{t("photoGuide.subtitle")}</p>
          </div>
          <button type="button" className="icon-btn" onClick={onClose}
                  aria-label={t("photoGuide.close")}>✕</button>
        </header>

        <div className="photo-guide-grid">
          {TIPS.map((tip) => (
            <article key={tip.key} className={`photo-guide-card guide-${tip.visual}`}>
              <div className="photo-guide-visual" aria-hidden="true">
                <span className="guide-emoji">{tip.icon}</span>
                <GuideSketch kind={tip.visual} />
              </div>
              <h4>{t(tip.titleKey)}</h4>
              <p>{t(tip.bodyKey)}</p>
            </article>
          ))}
        </div>

        <button type="button" className="btn btn-primary btn-large btn-full" onClick={onClose}>
          {t("photoGuide.close")}
        </button>
      </section>
    </div>
  );
}

function GuideSketch({ kind }) {
  if (kind === "daylight") {
    return (
      <svg viewBox="0 0 120 64" className="guide-sketch" aria-hidden="true">
        <circle cx="30" cy="26" r="10" fill="currentColor" opacity=".85" />
        <path d="M42 26h30" stroke="currentColor" strokeWidth="2" strokeDasharray="3 4" />
        <rect x="74" y="30" width="34" height="20" rx="3" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
    );
  }
  if (kind === "single") {
    return (
      <svg viewBox="0 0 120 64" className="guide-sketch" aria-hidden="true">
        <rect x="14" y="6" width="92" height="52" rx="6" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="4 4" />
        <ellipse cx="60" cy="36" rx="22" ry="12" fill="currentColor" opacity=".85" />
        <circle cx="38" cy="28" r="7" fill="currentColor" opacity=".85" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 120 64" className="guide-sketch" aria-hidden="true">
      <path d="M8 52h104" stroke="currentColor" strokeWidth="2" opacity=".4" />
      <ellipse cx="62" cy="36" rx="26" ry="14" fill="currentColor" opacity=".85" />
      <circle cx="36" cy="26" r="8" fill="currentColor" opacity=".85" />
      <path d="M30 20l-4-10M42 20l4-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <path d="M84 34v14" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
