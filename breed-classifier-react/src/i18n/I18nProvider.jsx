/**
 * Backwards-compatible entry point.
 *
 * Older components import `{ I18nProvider, useI18n }` from this file; the
 * implementation now lives in LanguageContext.jsx so there is a single
 * reactive source of truth for language, dynamic localization and TTS.
 */
export {
  LanguageProvider as I18nProvider,
  LanguageProvider,
  useLanguage as useI18n,
  useLanguage,
  LANGUAGES,
} from "./LanguageContext.jsx";
