/**
 * Backwards-compatible wrapper. The 4-theme control now lives in
 * ThemeSwitcher.jsx; existing pages keep importing ThemeToggle.
 */
export { THEMES, applyStoredTheme, getInitialTheme } from "./ThemeSwitcher.jsx";
export { default } from "./ThemeSwitcher.jsx";
