import React from "react";
import { createRoot } from "react-dom/client";
import AppRouter from "./router.jsx";
import { I18nProvider } from "./i18n/I18nProvider.jsx";
import { applyStoredTheme } from "./components/ThemeSwitcher.jsx";
import "./styles.css";
import "./styles/themes.css";

// Apply the persisted theme before the first paint to avoid a flash of the
// default background when someone chose the daylight or harvest theme.
applyStoredTheme();

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <I18nProvider>
      <AppRouter />
    </I18nProvider>
  </React.StrictMode>
);
