import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { compareTraits, getBreedTraits } from "../data/breedTraits.js";

/**
 * Shared "Why this breed?" + top-3 look-alike comparison.
 * Used by the user result card and the legacy single-image panel so both
 * surfaces explain the same prediction in the same way.
 */
export default function TraitComparison({ breeds, defaultOpen = false }) {
  const { t } = useI18n();
  // null = follow the caller's tier default; a click then takes over.
  const [open, setOpen] = useState(null);
  const breedKey = (breeds || []).join("|");

  // A new prediction should re-apply the tier default (tier 2 = open).
  useEffect(() => {
    setOpen(null);
  }, [breedKey]);

  if (!breeds || breeds.length === 0) return null;

  const isOpen = open === null ? defaultOpen : open;
  const comparison = compareTraits(breeds.slice(0, 3));
  const topTraits = getBreedTraits(breeds[0]);
  const hasTraits = comparison.breeds.some((b) => b.traits);

  return (
    <div className="trait-explainer">
      <button
        type="button"
        className="btn-text-action trait-toggle"
        onClick={() => setOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <span aria-hidden="true">{isOpen ? "▾" : "▸"}</span>
        {t("result.why")} · {t("result.lookAlike")}
      </button>

      {isOpen && (
        <div className="trait-body">
          {topTraits?.keyDistinguisher && (
            <p className="trait-distinguisher">{topTraits.keyDistinguisher}</p>
          )}

          {hasTraits ? (
            <div className="trait-table-wrap">
              <table className="trait-table">
                <thead>
                  <tr>
                    <th>{t("result.trait")}</th>
                    {comparison.breeds.map((b) => (
                      <th key={b.key}>{b.name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {comparison.rows.map((row) => (
                    <tr key={row.key} className={row.differs ? "trait-row-differs" : ""}>
                      <th scope="row">{row.label}</th>
                      {row.values.map((value, i) => (
                        <td key={`${row.key}-${i}`}>{value || "—"}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="trait-legend">◆ {t("result.traitDiffers")}</p>
              <p className="trait-disclaimer">{t("result.traitDisclaimer")}</p>
            </div>
          ) : (
            <p className="trait-missing">{t("result.traitMissing")}</p>
          )}
        </div>
      )}
    </div>
  );
}
