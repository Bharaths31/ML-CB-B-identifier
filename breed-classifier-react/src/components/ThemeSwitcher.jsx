import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";

export const THEMES = [
  {
    id: "field-parchment",
    name: "Field Parchment",
    desc: "Warm daylight editorial",
    canvas: "#F4F1EA",
    accent: "#2A4D3E",
  },
  {
    id: "obsidian-loam",
    name: "Obsidian Loam",
    desc: "Warm dark barn / lab",
    canvas: "#121110",
    accent: "#D48C46",
  },
  {
    id: "clinical-slate",
    name: "Clinical Slate",
    desc: "Veterinary daylight instrument",
    canvas: "#EEF1F4",
    accent: "#0F4C5C",
  },
  {
    id: "night-telemetry",
    name: "Night Telemetry",
    desc: "Naval-grade dark instrumentation",
    canvas: "#0B0F14",
    accent: "#38BDF8",
  },
];

const STORAGE_KEY = "agri_theme";
const LEGACY_KEY = "breed-classifier-theme";
const DEFAULT_THEME = "field-parchment";

const LEGACY_MAP = {
  dark: "obsidian-loam",
  light: "field-parchment",
  emerald: "field-parchment",
  sunset: "obsidian-loam",
  cyber: "night-telemetry",
  "emerald-obsidian": "field-parchment",
  "meadow-daylight": "clinical-slate",
  "earthy-harvest": "obsidian-loam",
  "aqua-pasture": "night-telemetry",
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
