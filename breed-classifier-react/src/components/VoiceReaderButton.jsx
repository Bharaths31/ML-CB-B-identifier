import { useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";

/**
 * Reads the localized prediction aloud using speechSynthesis.
 *
 * The spoken sentence is built from the *translated* species, breed, gender,
 * age and milk-yield strings, and the voice is selected for ta-IN / hi-IN /
 * en-IN so Tamil and Hindi are not read with an English accent.
 */
export default function VoiceReaderButton({ result, advisory }) {
  const { t, lang, tBreed, tDynamic, speakText, stopSpeaking } = useI18n();
  const [speaking, setSpeaking] = useState(false);
  const mounted = useRef(true);

  const supported =
    typeof window !== "undefined" && "speechSynthesis" in window;

  const text = useMemo(
    () => buildScript({ result, advisory, t, tBreed, tDynamic }),
    [result, advisory, t, tBreed, tDynamic]
  );

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      stopSpeaking();
    };
  }, [stopSpeaking]);

  // Switching language must stop an in-flight English readout.
  useEffect(() => {
    stopSpeaking();
    setSpeaking(false);
  }, [lang, stopSpeaking]);

  if (!supported) {
    return (
      <button type="button" className="btn btn-outline voice-btn" disabled
              title={t("result.noSpeech")}>
        🔊 {t("result.listen")}
      </button>
    );
  }

  const handleClick = () => {
    if (speaking) {
      stopSpeaking();
      setSpeaking(false);
      return;
    }
    const started = speakText(text, {
      onEnd: () => {
        if (mounted.current) setSpeaking(false);
      },
    });
    if (started) setSpeaking(true);
  };

  return (
    <button
      type="button"
      className={`btn btn-outline voice-btn ${speaking ? "is-speaking" : ""}`}
      onClick={handleClick}
      aria-pressed={speaking}
      title={t("result.listen")}
    >
      <span aria-hidden="true">{speaking ? "⏹" : "🔊"}</span>
      {speaking ? t("result.listening") : t("result.listen")}
    </button>
  );
}

function buildScript({ result, advisory, t, tBreed, tDynamic }) {
  if (!result || result.error) return "";
  const species = tDynamic("species", result.predicted_species || result.species);
  const breed = tBreed(result.predicted_breed || result.top_breed);
  const genderKey = result.gender?.key;
  const ageKey = result.age_group?.key;
  const gender = genderKey ? tDynamic("gender", genderKey) : "";
  const age = ageKey ? tDynamic("age", ageKey) : "";
  const milk =
    advisory?.production?.milk?.daily_yield ||
    result.knowledge?.production?.milk?.daily_yield;

  return [
    species ? `${t("hero.species")}: ${species}.` : "",
    breed ? `${t("hero.breed")}: ${breed}.` : "",
    gender ? `${t("hero.gender")}: ${gender}.` : "",
    age ? `${t("hero.age")}: ${age}.` : "",
    milk ? `${t("advisory.dailyYield")}: ${milk}.` : "",
  ].filter(Boolean).join(" ");
}
