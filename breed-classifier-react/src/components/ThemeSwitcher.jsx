import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";

export const THEMES = [
  {
    id: "emerald-obsidian",
    nameKey: "theme.emeraldObsidian",
    descKey: "theme.emeraldObsidian.desc",
    swatch: "linear-gradient(135deg, #10b981 0%, #121614 100%)",
    bg: "#121614",
  },
  {
    id: "meadow-daylight",
    nameKey: "theme.meadowDaylight",
    descKey: "theme.meadowDaylight.desc",
    swatch: "linear-gradient(135deg, #10b981 0%, #f4fbf7 100%)",
    bg: "#f4fbf7",
  },
  {
    id: "earthy-harvest",
    nameKey: "theme.earthyHarvest",
    descKey: "theme.earthyHarvest.desc",
    swatch: "linear-gradient(135deg, #84cc16 0%, #d97706 100%)",
    bg: "#181614",
  },
  {
    id: "aqua-pasture",
    nameKey: "theme.aquaPasture",
    descKey: "theme.aquaPasture.desc",
    swatch: "linear-gradient(135deg, #2dd4bf 0%, #11171a 100%)",
    bg: "#11171a",
  },
];

const STORAGE_KEY = "agri_theme";
const LEGACY_KEY = "breed-classifier-theme";
const DEFAULT_THEME = "emerald-obsidian";

export function getInitialTheme() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_KEY);
    if (THEMES.some((theme) => theme.id === stored)) return stored;
  } catch {
    /* storage may be blocked */
  }
  return DEFAULT_THEME;
}

export function applyStoredTheme() {
  const theme = getInitialTheme();
  document.documentElement.setAttribute("data-theme", theme);
  return theme;
}

/** Compact 4-swatch pill for the navbar. */
export default function ThemeSwitcher() {
  const { t } = useI18n();
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  return (
    <div className="theme-switcher" role="group" aria-label={t("theme.label")}>
      <span className="theme-switcher-label">{t("theme.label")}</span>
      {THEMES.map((item) => (
        <button
          key={item.id}
          type="button"
          className={`theme-switcher-btn ${theme === item.id ? "is-active" : ""}`}
          onClick={() => setTheme(item.id)}
          aria-pressed={theme === item.id}
          title={`${t(item.nameKey)} — ${t(item.descKey)}`}
        >
          <span className="theme-switcher-swatch" style={{ background: item.swatch }} />
        </button>
      ))}
    </div>
  );
}
