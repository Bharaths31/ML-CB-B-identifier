import { useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";

const LOCALES = { en: "en-IN", ta: "ta-IN", hi: "hi-IN" };

/**
 * Reads the prediction aloud with the browser's native speech synthesis.
 * Locale follows the UI language (en-IN / ta-IN / hi-IN). No dependencies.
 */
export default function VoiceReaderButton({ result, advisory, className = "" }) {
  const { t, lang } = useI18n();
  const [speaking, setSpeaking] = useState(false);
  const utteranceRef = useRef(null);

  const supported =
    typeof window !== "undefined" && "speechSynthesis" in window;

  const text = useMemo(() => buildScript(result, advisory, t), [result, advisory, t]);

  useEffect(() => () => {
    if (supported) window.speechSynthesis.cancel();
  }, [supported]);

  const stop = () => {
    if (!supported) return;
    window.speechSynthesis.cancel();
    setSpeaking(false);
  };

  const speak = () => {
    if (!supported || !text) return;
    if (speaking) {
      stop();
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = LOCALES[lang] || "en-IN";
    utterance.rate = 0.95;
    utterance.pitch = 1;
    const voices = window.speechSynthesis.getVoices();
    const match =
      voices.find((v) => v.lang === utterance.lang) ||
      voices.find((v) => v.lang?.startsWith(lang));
    if (match) utterance.voice = match;
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    utteranceRef.current = utterance;
    setSpeaking(true);
    window.speechSynthesis.speak(utterance);
  };

  if (!supported) {
    return (
      <button type="button" className={`btn btn-outline voice-btn ${className}`} disabled
              title={t("result.noSpeech")}>
        🔊 {t("result.listen")}
      </button>
    );
  }

  return (
    <button
      type="button"
      className={`btn btn-outline voice-btn ${speaking ? "is-speaking" : ""} ${className}`}
      onClick={speak}
      aria-pressed={speaking}
      title={t("result.listen")}
    >
      <span aria-hidden="true">{speaking ? "⏹" : "🔊"}</span>
      {speaking ? t("result.listening") : t("result.listen")}
    </button>
  );
}

function buildScript(result, advisory, t) {
  if (!result || result.error) return "";
  const species = result.predicted_species || result.species || "";
  const breed = (result.predicted_breed || result.top_breed || "").replace(/_/g, " ");
  const gender = result.gender?.key || result.gender?.short_label || "";
  const age = result.age_group?.key || result.age_group?.short_label || "";
  const genderWord = gender ? t(`advisory.${String(gender).toLowerCase()}`) : "";
  const ageWord = age ? t(`advisory.${String(age).toLowerCase()}`) : "";
  const milk = advisory?.production?.milk?.daily_yield || result.knowledge?.production?.milk?.daily_yield;

  const parts = [
    `${t("hero.species")}: ${species}.`,
    breed ? `${t("hero.breed")}: ${breed}.` : "",
    genderWord ? `${t("hero.gender")}: ${genderWord}.` : "",
    ageWord ? `${t("hero.age")}: ${ageWord}.` : "",
    milk ? `${t("advisory.dailyYield")}: ${milk}.` : "",
  ];
  return parts.filter(Boolean).join(" ");
}
