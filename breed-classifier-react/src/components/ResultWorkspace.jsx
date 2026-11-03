import { useEffect, useMemo, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import HeroFourGrid from "./HeroFourGrid.jsx";
import TraitChecklist from "./TraitChecklist.jsx";
import FarmerAdvisoryCard from "./FarmerAdvisoryCard.jsx";
import BreedEncyclopedia from "./BreedEncyclopedia.jsx";
import PhotoGuidance from "./PhotoGuidance.jsx";
import VoiceReaderButton from "./VoiceReaderButton.jsx";
import { getBreedKnowledge, recalculateAdvisory } from "../api.js";
import { downloadFarmerReport } from "../utils/reportCard.js";

const cap = (value) => (value ? value.charAt(0).toUpperCase() + value.slice(1) : null);

/**
 * User-portal results workspace.
 *
 * Owns the gender/age overrides so the Big-4 grid and the advisory stay in
 * sync: changing a pill calls POST /api/advisory-recalculate (~0.1 ms) and
 * never re-runs the image model.
 */
export default function ResultWorkspace({
  result,
  quality,
  previewUrl,
  filename,
  onReset,
  onRetake,
  onOpenAdvanced,
}) {
  const { t, lang, tBreed, tDynamic } = useI18n();
  const isOod = Boolean(result?.is_ood ?? result?.ood);

  const [gender, setGender] = useState(null);
  const [ageGroup, setAgeGroup] = useState(null);
  const [advisory, setAdvisory] = useState(null);
  const [latency, setLatency] = useState(null);
  const [busy, setBusy] = useState(false);
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

  // A new prediction resets every override.
  useEffect(() => {
    setGender(result?.gender?.key || null);
    setAgeGroup(result?.age_group?.key || null);
    setAdvisory(result?.knowledge || null);
    setLatency(null);
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

  const refreshAdvisory = async (nextGender, nextAge) => {
    if (!species || !breedSlug) return;
    setBusy(true);
    setNotice("");
    try {
      const data = await recalculateAdvisory({
        species,
        breed: breedSlug,
        gender: cap(nextGender),
        ageGroup: cap(nextAge),
      });
      setAdvisory(data.advisory);
      setLatency(data.latency_ms);
    } catch (err) {
      setNotice(err?.message || "Advisory refresh failed");
    } finally {
      setBusy(false);
    }
  };

  const handleGender = (value) => {
    setGender(value);
    refreshAdvisory(value, ageGroup);
  };
  const handleAge = (value) => {
    setAgeGroup(value);
    refreshAdvisory(gender, value);
  };

  const reportLabels = useMemo(() => ({
    "report.title": t("result.downloadCard"),
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

  const overridden = Boolean(
    (gender && result?.gender?.key && gender !== result.gender.key) ||
    (ageGroup && result?.age_group?.key && ageGroup !== result.age_group.key)
  );

  return (
    <section className="result-workspace">
      <header className="workspace-head">
        <div className="workspace-title">
          <span className="workspace-kicker">
            {isOod ? t("result.oodBadge") : t("result.predictedBreed")}
          </span>
          <h2>{isOod ? t("result.oodTitle") : localizedBreed || t("hero.unknown")}</h2>
        </div>
        <div className="workspace-actions">
          <VoiceReaderButton result={result} advisory={advisory} />
          <button type="button" className="btn btn-outline" onClick={handleDownload}>
            📥 {t("result.downloadCard")}
          </button>
          <button type="button" className="btn btn-outline" onClick={onOpenAdvanced}>
            ⚙ {t("advanced.openDetails")}
          </button>
        </div>
      </header>

      <PhotoGuidance result={result} />

      <HeroFourGrid
        result={result}
        gender={gender}
        ageGroup={ageGroup}
        onGenderChange={handleGender}
        onAgeChange={handleAge}
        breedKnowledge={entry}
      />

      {overridden && (
        <p className="workspace-override-note">
          {t("hero.verify")}:{" "}
          {[gender ? t(`advisory.${gender}`) : null,
            ageGroup ? t(`advisory.${ageGroup}`) : null,
            busy ? t("result.recalculating") : null]
            .filter(Boolean)
            .join(" · ")}
        </p>
      )}

      <TraitChecklist traits={result?.traits} />

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
          <FarmerAdvisoryCard advisory={advisory} loading={busy} latencyMs={latency} />
        ) : (
          <BreedEncyclopedia
            entry={entry}
            summary={result?.knowledge?.breed || null}
            loading={entryLoading}
          />
        )}
      </div>

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
