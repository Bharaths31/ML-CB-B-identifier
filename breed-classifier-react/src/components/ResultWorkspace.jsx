import { useEffect, useMemo, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import HeroFourGrid from "./HeroFourGrid.jsx";
import FarmerAdvisoryCard from "./FarmerAdvisoryCard.jsx";
import BreedEncyclopedia from "./BreedEncyclopedia.jsx";
import PhotoGuidance from "./PhotoGuidance.jsx";
import VoiceReaderButton from "./VoiceReaderButton.jsx";
import { getBreedKnowledge } from "../api.js";
import { downloadFarmerReport } from "../utils/reportCard.js";

/**
 * User-portal results workspace.
 *
 * Gender and age are read-only model estimates; the advisory comes from the
 * prediction response, so no manual override or recalculation is exposed.
 */
export default function ResultWorkspace({
  result,
  quality,
  previewUrl,
  filename,
  display = {},
  visibility = {},
  onReset,
  onRetake,
  onOpenAdvanced,
}) {
  const { t, lang, tBreed, tDynamic } = useI18n();
  const isOod = Boolean(result?.is_ood ?? result?.ood);
  const show = (key) => visibility[key] !== false;
  const showSpeciesBadge = display.show_species_badge !== false;
  const showSpeciesConfidence = display.show_species_confidence !== false;

  const [advisory, setAdvisory] = useState(null);
  const [notice, setNotice] = useState("");
  const [tab, setTab] = useState("farmer");
  const [entry, setEntry] = useState(null);
  const [entryLoading, setEntryLoading] = useState(false);

  const species = result?.predicted_species || result?.species;
  const breedSlug =
    result?.knowledge?.breed?.slug || result?.top5_breeds?.[0]?.breed_slug || null;
  const breedName = result?.predicted_breed || result?.top_breed || result?.knowledge?.breed?.display_name || "";
  const localizedBreed = tBreed(breedName);

  const localizedResult = useMemo(() => {
    if (!result) return result;
    return {
      ...result,
      predicted_breed: localizedBreed,
      species: tDynamic("species", result.predicted_species || result.species),
      gender: {
        ...(result.gender || {}),
        label: tDynamic("gender", result.gender?.key),
      },
      age_group: {
        ...(result.age_group || {}),
        label: tDynamic("age", result.age_group?.key),
      },
    };
  }, [result, localizedBreed, tDynamic]);

  // A new prediction resets the advisory and scroll state.
  useEffect(() => {
    setAdvisory(result?.knowledge || null);
    setNotice("");
    setTab("farmer");
  }, [result]);

  // Native-tract / encyclopedia entry (small JSON, cached server-side).
  useEffect(() => {
    if (!species || !breedSlug) {
      setEntry(null);
      return undefined;
    }
    let cancelled = false;
    setEntryLoading(true);
    getBreedKnowledge(species, breedSlug)
      .then((data) => {
        if (!cancelled) setEntry(data.knowledge);
      })
      .catch(() => {
        if (!cancelled) setEntry(null);
      })
      .finally(() => {
        if (!cancelled) setEntryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [species, breedSlug]);

  const reportLabels = useMemo(() => ({
    "report.title": t("report.title"),
    "hero.species": t("hero.species"),
    "hero.breed": t("hero.breed"),
    "hero.gender": t("hero.gender"),
    "hero.age": t("hero.age"),
    "hero.origin": t("hero.origin"),
    "advisory.production": t("advisory.production"),
    "advisory.dailyYield": t("advisory.dailyYield"),
    "advisory.lactation": t("advisory.lactation"),
    "advisory.fat": t("advisory.fat"),
    "advisory.nutrition": t("advisory.nutrition"),
    "advisory.greenFodder": t("advisory.greenFodder"),
    "advisory.dryFodder": t("advisory.dryFodder"),
    "advisory.concentrate": t("advisory.concentrate"),
    "advisory.minerals": t("advisory.minerals"),
    "advisory.actions": t("advisory.actions"),
    "advisory.schemes": t("advisory.schemes"),
    "advisory.disclaimer": t("advisory.disclaimer"),
  }), [t]);

  const handleDownload = () => {
    const opened = downloadFarmerReport({
      result: localizedResult,
      advisory,
      quality,
      imageUrl: previewUrl,
      filename: filename || result?.filename,
      labels: reportLabels,
      lang,
    });
    setNotice(opened ? t("result.reportReady") : t("result.reportBlocked"));
    setTimeout(() => setNotice(""), 5000);
  };

  return (
    <section className="result-workspace">
      <header className="workspace-head">
        <div className="workspace-title">
          <span className="workspace-kicker">
            {isOod ? t("result.oodBadge") : t("result.predictedBreed")}
          </span>
          <h2>{isOod ? t("result.oodTitle") : localizedBreed || t("hero.unknown")}</h2>
          {showSpeciesBadge && !isOod && (
            <span className={`badge badge-neutral workspace-species-badge species-${String(result?.predicted_species || result?.species || "cattle").toLowerCase()}`}>
              {tDynamic("species", result?.predicted_species || result?.species)}
            </span>
          )}
        </div>
        <div className="workspace-actions">
          {show("voice") && <VoiceReaderButton result={result} advisory={advisory} />}
          {show("reportCard") && (
            <button type="button" className="btn btn-outline" onClick={handleDownload}>
              📥 {t("result.downloadCard")}
            </button>
          )}
          {show("advanced") && onOpenAdvanced && (
            <button type="button" className="btn btn-outline" onClick={onOpenAdvanced}>
              ⚙ {t("advanced.openDetails")}
            </button>
          )}
        </div>
      </header>

      {show("photoGuidance") && <PhotoGuidance result={result} />}

      <HeroFourGrid
        result={result}
        breedKnowledge={entry}
        showSpeciesConfidence={showSpeciesConfidence}
      />

      {show("advisory") && (
        <>
          <div className="result-tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={tab === "farmer"}
              className={`result-tab ${tab === "farmer" ? "is-active" : ""}`}
              onClick={() => setTab("farmer")}
            >
              {t("result.tabs.farmer")}
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === "public"}
              className={`result-tab ${tab === "public" ? "is-active" : ""}`}
              onClick={() => setTab("public")}
            >
              {t("result.tabs.public")}
            </button>
          </div>

          <div className="result-tab-body">
            {tab === "farmer" ? (
              <FarmerAdvisoryCard advisory={advisory} loading={false} latencyMs={null} />
            ) : (
              <BreedEncyclopedia
                entry={entry}
                summary={result?.knowledge?.breed || null}
                loading={entryLoading}
              />
            )}
          </div>
        </>
      )}

      {notice && <p className="workspace-notice" role="status">{notice}</p>}

      <footer className="workspace-footer">
        {onRetake && !isOod && (
          <button type="button" className="btn btn-secondary" onClick={onRetake}>
            📷 {t("result.retakeCoach")}
          </button>
        )}
        <button type="button" className="btn btn-primary" onClick={onReset}>
          {t("result.classifyAnother")}
        </button>
      </footer>
    </section>
  );
}
