import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { TRAIT_FIELDS } from "../data/breedTraits.js";

/**
 * Explainable-AI checklist: the model's morphological traits as chips the
 * farmer can confirm or reject. Purely client-side; no extra request.
 */
export default function TraitChecklist({ traits }) {
  const { t } = useI18n();
  const fields = TRAIT_FIELDS.filter((field) => traits?.[field.key]);
  const [checked, setChecked] = useState({});

  useEffect(() => {
    setChecked({});
  }, [traits]);

  if (fields.length === 0) return null;

  const matched = fields.filter((field) => checked[field.key]).length;
  const allMatch = matched === fields.length;

  const toggle = (key) =>
    setChecked((prev) => ({ ...prev, [key]: !prev[key] }));

  return (
    <section className="trait-checklist">
      <header className="trait-checklist-head">
        <div>
          <h3>{t("traitCheck.title")}</h3>
          <p>{t("traitCheck.prompt")}</p>
        </div>
        <span className={`trait-checklist-count ${allMatch ? "is-complete" : ""}`}>
          {t("traitCheck.matched", { matched, total: fields.length })}
        </span>
      </header>

      <div className="trait-chip-grid">
        {fields.map((field) => {
          const isOn = Boolean(checked[field.key]);
          return (
            <label key={field.key} className={`trait-chip ${isOn ? "trait-chip-on" : ""}`}>
              <input
                type="checkbox"
                checked={isOn}
                onChange={() => toggle(field.key)}
              />
              <span className="trait-chip-check" aria-hidden="true">{isOn ? "✓" : "+"}</span>
              <span className="trait-chip-text">
                <span className="trait-chip-label">{field.label}</span>
                <span className="trait-chip-value">{traits[field.key]}</span>
              </span>
            </label>
          );
        })}
      </div>

      <div className="trait-checklist-footer">
        <p className={`trait-checklist-note ${allMatch ? "is-complete" : ""}`}>
          {allMatch ? t("traitCheck.good") : t("traitCheck.partial")}
        </p>
        {matched > 0 && (
          <button
            type="button"
            className="btn-text-action"
            onClick={() => setChecked({})}
          >
            {t("traitCheck.reset")}
          </button>
        )}
      </div>
    </section>
  );
}
