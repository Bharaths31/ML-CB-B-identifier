import React from "react";
import { createRoot } from "react-dom/client";
import { I18nProvider } from "../../breed-classifier-react/src/i18n/I18nProvider.jsx";
import { applyStoredTheme } from "../../breed-classifier-react/src/components/ThemeSwitcher.jsx";
import ExpertPage from "./ExpertPage.jsx";
import "../../breed-classifier-react/src/styles.css";
import "../../breed-classifier-react/src/styles/themes.css";

applyStoredTheme();

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <I18nProvider>
      <ExpertPage />
    </I18nProvider>
  </React.StrictMode>
);
