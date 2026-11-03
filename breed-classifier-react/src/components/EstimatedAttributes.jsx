import { useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { TRAIT_FIELDS } from "../data/breedTraits.js";

/**
 * Post-hoc gender / age / morphological traits from the enriched /api/predict
 * payload. Renders nothing for older backends that do not send these blocks.
 *
 * The heuristic engine reports `requires_confirmation`, so this is the one-tap
 * farmer confirmation the API asks for. The choice is local for now (no
 * persistence endpoint yet).
 */
export default function EstimatedAttributes({ result }) {
  const { t } = useI18n();
  const [confirmed, setConfirmed] = useState(false);

  const gender = result?.gender || {};
  const age = result?.age_group || {};
  const traits = result?.traits || {};

  const traitFields = TRAIT_FIELDS.filter((field) => traits[field.key]);
  const needsConfirm = Boolean(gender.requires_confirmation || age.requires_confirmation);
  const method = [gender.method, age.method].find(
    (m) => m && m !== "skipped"
  );
  const methodLabel =
    method === "clip_zeroshot"
      ? t("attributes.method.clip")
      : method === "morphological_heuristic"
      ? t("attributes.method.heuristic")
      : null;

  const hasAnything =
    gender.label || age.label || traitFields.length > 0;
  if (!hasAnything) return null;

  return (
    <section className="advanced-result-block">
      <h4>{t("attributes.title")}</h4>

      <div className="advanced-quality-grid">
        {gender.label && (
          <div>
            <span>{t("attributes.gender")}</span>
            <strong>
              {gender.short_label || gender.label}
              {typeof gender.confidence === "number"
                ? ` · ${Math.round(gender.confidence)}%`
                : ""}
            </strong>
          </div>
        )}
        {age.label && (
          <div>
            <span>{t("attributes.age")}</span>
            <strong>
              {age.short_label || age.label}
              {typeof age.confidence === "number"
                ? ` · ${Math.round(age.confidence)}%`
                : ""}
            </strong>
          </div>
        )}
      </div>

      {traitFields.length > 0 && (
        <>
          <h4 className="section-subtitle">{t("attributes.traits")}</h4>
          <div className="advanced-quality-grid">
            {traitFields.map((field) => (
              <div key={field.key}>
                <span>{field.label}</span>
                <strong>{traits[field.key]}</strong>
              </div>
            ))}
          </div>
        </>
      )}

      {methodLabel && (
        <p className="advanced-result-note">
          {t("attributes.method")}: {methodLabel}
        </p>
      )}

      {needsConfirm && (
        <>
          <p className="advanced-result-note">{t("attributes.heuristicNote")}</p>
          <button
            type="button"
            className="btn btn-sm btn-outline"
            onClick={() => setConfirmed(true)}
            disabled={confirmed}
          >
            {confirmed ? t("attributes.confirmed") : t("attributes.confirm")}
          </button>
        </>
      )}

      {!needsConfirm && method === "clip_zeroshot" && (
        <p className="advanced-result-note">{t("attributes.clipNote")}</p>
      )}
    </section>
  );
}
