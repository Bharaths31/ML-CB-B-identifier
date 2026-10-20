import React from "react";
import { createRoot } from "react-dom/client";
import AppRouter from "./router.jsx";
import { I18nProvider } from "./i18n/I18nProvider.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <I18nProvider>
      <AppRouter />
    </I18nProvider>
  </React.StrictMode>
);

