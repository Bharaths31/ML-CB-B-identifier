import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { getStoredLanguage, STORAGE_KEY, TRANSLATIONS, translate } from "./translations.js";

const I18nContext = createContext({
  lang: "en",
  setLang: () => {},
  t: (key) => key,
});

export function I18nProvider({ children }) {
  const [lang, setLangState] = useState(getStoredLanguage);

  const setLang = useCallback((next) => {
    if (!TRANSLATIONS[next]) return;
    setLangState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("lang", lang === "ta" ? "ta-IN" : "en");
  }, [lang]);

  const t = useCallback((key, vars) => translate(lang, key, vars), [lang]);

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  return useContext(I18nContext);
}
