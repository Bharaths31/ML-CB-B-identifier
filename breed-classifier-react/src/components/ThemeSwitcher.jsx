import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";

export const THEMES = [
  {
    id: "emerald-obsidian",
    name: "Emerald Obsidian",
    desc: "Fresh green & light black",
    canvas: "#121714",
    accent: "#3FA66B",
  },
  {
    id: "meadow-daylight",
    name: "Meadow Daylight",
    desc: "Fresh meadow daylight",
    canvas: "#F4F1EA",
    accent: "#2F7D4A",
  },
  {
    id: "earthy-harvest",
    name: "Earthy Harvest",
    desc: "Golden harvest & charcoal",
    canvas: "#1C1813",
    accent: "#D9A441",
  },
  {
    id: "aqua-pasture",
    name: "Aqua Pasture",
    desc: "Monsoon breeze",
    canvas: "#EEF4F4",
    accent: "#0F6F73",
  },
];

const STORAGE_KEY = "agri_theme";
const LEGACY_KEY = "breed-classifier-theme";
const DEFAULT_THEME = "emerald-obsidian";

const LEGACY_MAP = {
  dark: "emerald-obsidian",
  light: "meadow-daylight",
  emerald: "emerald-obsidian",
  sunset: "earthy-harvest",
  cyber: "emerald-obsidian",
  "field-parchment": "meadow-daylight",
  "obsidian-loam": "emerald-obsidian",
  "clinical-slate": "aqua-pasture",
  "night-telemetry": "emerald-obsidian",
};

export function getInitialTheme() {
  try {
    const stored =
      localStorage.getItem(STORAGE_KEY) || localStorage.getItem(LEGACY_KEY);
    if (THEMES.some((theme) => theme.id === stored)) return stored;
    if (LEGACY_MAP[stored]) return LEGACY_MAP[stored];
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

/** Compact 4-swatch pill: each circle is split canvas/accent at 45°. */
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
          title={`${item.name} — ${item.desc}`}
        >
          <span
            className="theme-switcher-swatch"
            style={{
              background: `linear-gradient(135deg, ${item.canvas} 0 50%, ${item.accent} 50% 100%)`,
            }}
          />
        </button>
      ))}
    </div>
  );
}
