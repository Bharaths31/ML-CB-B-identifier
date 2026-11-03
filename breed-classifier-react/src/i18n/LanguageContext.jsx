import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { LANGUAGES, TRANSLATIONS, translate } from "./translations.js";
import {
  BREED_NAMES_I18N,
  DYNAMIC_I18N,
  TRAIT_FIELDS_I18N,
  localizeAdvisory,
  localizeBreed,
  localizeDynamic,
  localizeTraitValue,
} from "./dictionaries.js";

const STORAGE_KEY = "agri_lang";
const LEGACY_KEY = "breed-classifier-lang";
const LOCALES = { en: "en-IN", ta: "ta-IN", hi: "hi-IN" };

const LanguageContext = createContext({
  lang: "en",
  setLang: () => {},
  t: (key) => key,
  tBreed: (value) => value,
  tDynamic: (kind, value) => value,
  tTrait: (field) => field,
  tTraitValue: (field, value) => value,
  renderAdvisory: (advisory) => advisory,
  speakText: () => false,
  stopSpeaking: () => {},
  languages: LANGUAGES,
});

function readLang() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_KEY);
    if (stored && TRANSLATIONS[stored]) return stored;
  } catch {
    /* storage can be blocked in private mode */
  }
  return "en";
}

/**
 * Reactive language context.
 *
 * Every consumer re-renders when `lang` changes because the value object is
 * rebuilt in useMemo. Dynamic payload values (breeds, traits, species, gender,
 * age, advisory prose) go through the helpers below, so switching language
 * never requires a new API call.
 */
export function LanguageProvider({ children }) {
  const [lang, setLangState] = useState(readLang);

  useEffect(() => {
    document.documentElement.setAttribute("lang", lang);
    document.documentElement.setAttribute("data-lang", lang);
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch {
      /* ignore */
    }
  }, [lang]);

  const setLang = useCallback((next) => {
    if (TRANSLATIONS[next]) setLangState(next);
  }, []);

  const t = useCallback((key, vars) => translate(lang, key, vars), [lang]);

  const tBreed = useCallback((value) => localizeBreed(value, lang), [lang]);
  const tDynamic = useCallback((kind, value) => localizeDynamic(kind, value, lang), [lang]);
  const tTrait = useCallback((field) => {
    const entry = TRAIT_FIELDS_I18N[field];
    return entry ? entry[lang] || entry.en : field;
  }, [lang]);
  const tTraitValue = useCallback(
    (field, value) => localizeTraitValue(field, value, lang),
    [lang]
  );

  const renderAdvisory = useCallback(
    (advisory) => localizeAdvisory(advisory, lang, { tBreed, tDynamic }),
    [lang, tBreed, tDynamic]
  );

  const stopSpeaking = useCallback(() => {
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
  }, []);

  const speakText = useCallback(
    (text, handlers = {}) => {
      if (typeof window === "undefined" || !("speechSynthesis" in window)) return false;
      const clean = String(text || "").trim();
      if (!clean) return false;

      const synth = window.speechSynthesis;
      synth.cancel();
      const utterance = new SpeechSynthesisUtterance(clean);
      utterance.lang = LOCALES[lang] || "en-IN";
      utterance.rate = 0.92;
      utterance.pitch = 1;
      utterance.onend = () => handlers.onEnd?.();
      utterance.onerror = () => handlers.onEnd?.();

      const voices = synth.getVoices();
      const target = LOCALES[lang];
      const exact = voices.find((v) => v.lang?.replace("_", "-") === target);
      const near = voices.find((v) => v.lang?.startsWith(lang));
      if (exact || near) utterance.voice = exact || near;
      synth.speak(utterance);
      return true;
    },
    [lang]
  );

  const value = useMemo(
    () => ({
      lang,
      setLang,
      t,
      tBreed,
      tDynamic,
      tTrait,
      tTraitValue,
      renderAdvisory,
      speakText,
      stopSpeaking,
      languages: LANGUAGES,
      dictionary: TRANSLATIONS[lang] || TRANSLATIONS.en,
      breedNames: BREED_NAMES_I18N,
      dynamicTerms: DYNAMIC_I18N,
    }),
    [lang, setLang, t, tBreed, tDynamic, tTrait, tTraitValue, renderAdvisory,
     speakText, stopSpeaking]
  );

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  return useContext(LanguageContext);
}

export { LANGUAGES };
