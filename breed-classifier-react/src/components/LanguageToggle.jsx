import { useI18n } from "../i18n/I18nProvider.jsx";
import { LANGUAGES } from "../i18n/translations.js";

export default function LanguageToggle() {
  const { lang, setLang, t } = useI18n();

  return (
    <div className="lang-toggle" role="group" aria-label={t("header.language")}>
      {LANGUAGES.map((l) => (
        <button
          key={l.id}
          type="button"
          className={`lang-toggle-btn ${lang === l.id ? "lang-active" : ""}`}
          onClick={() => setLang(l.id)}
          title={l.label}
          aria-pressed={lang === l.id}
        >
          {l.short}
        </button>
      ))}
    </div>
  );
}
