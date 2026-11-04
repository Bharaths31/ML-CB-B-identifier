import { useEffect, useRef, useState } from "react";
import { THEMES } from "./ThemeSwitcher.jsx";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { LANGUAGES } from "../i18n/LanguageContext.jsx";
import {
  GlobeIcon,
  LanguageIcon,
  SlidersIcon,
  ThemeIcon,
} from "./AndroidIcons.jsx";
import { getLastBackendUrl } from "../services/apiBase.js";
import {
  DEFAULT_SETTINGS,
  deleteOodProfile,
  listOodProfiles,
  saveModelFile,
  saveOodProfile,
} from "../services/onDeviceInference.js";

const LANGUAGE_LABELS = {
  en: { native: "English", english: "English · EN" },
  ta: { native: "தமிழ்", english: "Tamil · TA" },
  hi: { native: "हिन्दी", english: "Hindi · HI" },
};

function ToggleRow({ label, help, checked, onChange }) {
  return (
    <div className="control-setting-row">
      <div className="setting-info">
        <span className="setting-label">{label}</span>
        {help && <span className="setting-subtext">{help}</span>}
      </div>
      <label className="toggle-switch">
        <input type="checkbox" checked={Boolean(checked)} onChange={(e) => onChange(e.target.checked)} />
        <span className="toggle-slider" />
      </label>
    </div>
  );
}

export default function SettingsModal({
  open,
  onClose,
  settings,
  models = [],
  oodProfiles = [],
  backendStatus = "offline",
  onSettingsChange,
  onModelsChanged,
  onOodChanged,
}) {
  const { lang, setLang, t } = useI18n();
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [online, setOnline] = useState(() => (typeof navigator === "undefined" ? true : navigator.onLine));
  const modelInputRef = useRef(null);
  const oodInputRef = useRef(null);
  const modelSelectRef = useRef(null);

  const current = settings || DEFAULT_SETTINGS;
  const fallbackBackendUrl = getLastBackendUrl();
  const backendConnected = Boolean(
    current.onlineMode && online && backendStatus === "connected"
  );
  const backendLabel = !current.onlineMode || !online
    ? t("android.offline")
    : backendStatus === "unconfigured"
      ? t("android.backendNotConfigured")
      : backendConnected
        ? t("header.connected")
        : t("header.reconnecting");
  const activeOod = oodProfiles.find((p) => p.filename === current.activeOodProfile);
  const activeModelMeta = models.find((m) => m.name === current.activeModel)?.metadata;
  const needsManualOod = Boolean(
    current.oodEnabled
    && current.activeOodProfile === "auto"
    && activeModelMeta?.source === "uploaded"
    && !activeModelMeta?.ood
  );

  useEffect(() => {
    const update = () => setOnline(navigator.onLine);
    window.addEventListener("online", update);
    window.addEventListener("offline", update);
    return () => {
      window.removeEventListener("online", update);
      window.removeEventListener("offline", update);
    };
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event) => { if (event.key === "Escape") onClose?.(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const update = async (patch) => {
    await onSettingsChange(patch);
  };

  const changeTheme = async (themeId) => {
    document.documentElement.setAttribute("data-theme", themeId);
    try { localStorage.setItem("agri_theme", themeId); } catch { /* ignore */ }
    await update({ theme: themeId });
  };

  const changeLanguage = async (next) => {
    setLang(next);
    await update({ language: next });
  };

  const handleModelUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setBusy(true);
    setMessage("");
    try {
      const record = await saveModelFile(file);
      await onModelsChanged?.();
      if (record.metadata?.runnable !== false) {
        const embedded = record.metadata?.ood;
        if (embedded) {
          await update({
            activeModel: record.name,
            activeOodProfile: "auto",
            oodEnabled: true,
            energyThreshold: Number(embedded.energy_threshold ?? current.energyThreshold),
            mspThreshold: Number(embedded.msp_threshold ?? current.mspThreshold),
            binaryConfidenceFloor: Number(embedded.binary_confidence_floor ?? current.binaryConfidenceFloor),
            energyTemperature: Number(embedded.energy_temperature ?? current.energyTemperature),
          });
          setMessage(t("android.modelAddedOod", { name: record.name }));
        } else {
          await update({
            activeModel: record.name,
            activeOodProfile: "auto",
            oodEnabled: false,
          });
          setMessage(t("android.modelAddedNoOod", { name: record.name }));
        }
      } else {
        setMessage(t("android.modelSavedPt", { name: record.name }));
      }
      requestAnimationFrame(() => {
        modelSelectRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
        modelSelectRef.current?.focus();
      });
    } catch (err) {
      setMessage(err.message || t("android.modelUploadFailed"));
    } finally {
      setBusy(false);
    }
  };

  const selectModel = async (name) => {
    const model = models.find((item) => item.name === name);
    const embedded = model?.metadata?.ood;
    if (embedded) {
      await update({
        activeModel: name,
        activeOodProfile: "auto",
        oodEnabled: true,
        energyThreshold: Number(embedded.energy_threshold ?? current.energyThreshold),
        mspThreshold: Number(embedded.msp_threshold ?? current.mspThreshold),
        binaryConfidenceFloor: Number(embedded.binary_confidence_floor ?? current.binaryConfidenceFloor),
        energyTemperature: Number(embedded.energy_temperature ?? current.energyTemperature),
      });
      setMessage(t("android.modelOodApplied", { name }));
    } else {
      await update({ activeModel: name, activeOodProfile: "auto", oodEnabled: false });
      setMessage(t("android.modelNoOod", { name }));
    }
  };

  const handleOodUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setBusy(true);
    setMessage("");
    try {
      const text = await file.text();
      const json = JSON.parse(text);
      const record = await saveOodProfile(file.name, json);
      await onOodChanged?.();
      await update({
        activeOodProfile: record.filename,
        oodEnabled: true,
        energyThreshold: record.json_data.energy_threshold,
        mspThreshold: record.json_data.msp_threshold,
        binaryConfidenceFloor: record.json_data.binary_confidence_floor,
        energyTemperature: record.json_data.energy_temperature,
      });
      setMessage(t("android.oodSaved", { file: record.filename }));
    } catch (err) {
      setMessage(err.message || t("android.oodUploadFailed"));
    } finally {
      setBusy(false);
    }
  };

  const selectOodProfile = async (value) => {
    if (value === "disabled") {
      await update({ activeOodProfile: "disabled", oodEnabled: false });
      return;
    }
    if (value === "auto") {
      await update({
        activeOodProfile: "auto",
        oodEnabled: true,
        energyThreshold: DEFAULT_SETTINGS.energyThreshold,
        mspThreshold: DEFAULT_SETTINGS.mspThreshold,
        binaryConfidenceFloor: DEFAULT_SETTINGS.binaryConfidenceFloor,
        energyTemperature: DEFAULT_SETTINGS.energyTemperature,
      });
      return;
    }
    const profile = oodProfiles.find((p) => p.filename === value);
    if (!profile) return;
    await update({
      activeOodProfile: value,
      oodEnabled: true,
      energyThreshold: profile.json_data.energy_threshold,
      mspThreshold: profile.json_data.msp_threshold,
      binaryConfidenceFloor: profile.json_data.binary_confidence_floor,
      energyTemperature: profile.json_data.energy_temperature,
    });
  };

  return (
    <div className="settings-overlay" role="presentation" onClick={onClose}>
      <aside
        className="settings-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={t("android.appSettings")}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="settings-head">
          <button
            type="button"
            className="settings-back"
            onClick={onClose}
            aria-label={t("actions.close")}
          >
            ←
          </button>
          <div className="settings-head-copy">
            <h2>{t("android.appSettings")}</h2>
            <p>{t("android.appSettingsSub")}</p>
          </div>
          <span className="settings-status-badge">
            <i aria-hidden="true" />
            {t("android.offlineReady")}
          </span>
        </header>

        {message && <div className="settings-message">{message}</div>}

        <section className="settings-section settings-language-section">
          <div className="settings-section-title">
            <span className="settings-section-icon" aria-hidden="true"><LanguageIcon /></span>
            <div>
              <h3>{t("android.selectLanguage")}</h3>
              <p>{t("android.languageDesc")}</p>
            </div>
          </div>
          <div className="settings-language-grid">
            {LANGUAGES.map((item) => {
              const active = lang === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  className={`settings-language-card ${active ? "is-active" : ""}`}
                  onClick={() => changeLanguage(item.id)}
                  aria-pressed={active}
                >
                  <span className="settings-language-code">{item.short}</span>
                  <span className="settings-language-copy">
                    <strong>{LANGUAGE_LABELS[item.id]?.native || item.label}</strong>
                    <small>{LANGUAGE_LABELS[item.id]?.english || item.label}</small>
                  </span>
                  <span className="settings-card-check" aria-hidden="true">
                    {active ? "✓" : ""}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="settings-section">
          <div className="settings-section-title">
            <span className="settings-section-icon" aria-hidden="true"><ThemeIcon /></span>
            <div>
              <h3>{t("android.themeDisplay")}</h3>
              <p>{t("android.themeDesc")}</p>
            </div>
          </div>
          <div className="settings-theme-grid">
            {THEMES.map((theme) => {
              const active = current.theme === theme.id;
              return (
                <button
                  key={theme.id}
                  type="button"
                  className={`settings-theme-btn ${active ? "is-active" : ""}`}
                  onClick={() => changeTheme(theme.id)}
                  aria-pressed={active}
                >
                  <span
                    className="settings-theme-swatch"
                    style={{ background: `linear-gradient(135deg, ${theme.canvas} 0 50%, ${theme.accent} 50% 100%)` }}
                  />
                  <span className="settings-theme-copy">
                    <strong>{theme.name}</strong>
                    <small>{t(`android.theme${theme.id === "emerald-obsidian" ? "Emerald" : theme.id === "meadow-daylight" ? "Meadow" : theme.id === "earthy-harvest" ? "Earthy" : "Aqua"}Desc`)}</small>
                  </span>
                  <span className="settings-card-check" aria-hidden="true">
                    {active ? "✓" : ""}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="settings-section">
          <div className="settings-section-title">
            <span className="settings-section-icon" aria-hidden="true"><GlobeIcon /></span>
            <div>
              <h3>{t("android.onlineMode")}</h3>
              <p>{t("android.onlineModeDesc")}</p>
            </div>
          </div>
          <ToggleRow
            label={t("android.onlineMode")}
            help={t("android.onlineModeHelp")}
            checked={current.onlineMode}
            onChange={(value) => update({ onlineMode: value })}
          />
          <div className={`settings-online-badge ${backendConnected ? "is-online" : "is-offline"}`}>
            <span className={`telemetry-dot ${backendConnected ? "online" : "offline"}`} />
            {backendLabel}
            <span className="settings-online-extra">
              {current.onlineMode ? t("android.onlineModeOn") : t("android.onlineModeOff")}
            </span>
          </div>
          <label className="settings-field">
            <span>{t("android.backendUrl")}</span>
            <input
              type="url"
              className="form-input"
              value={current.backendUrl || ""}
              placeholder={fallbackBackendUrl || "https://your-backend.example.com"}
              onChange={(e) => update({ backendUrl: e.target.value })}
            />
          </label>
          <p className="settings-hint">
            {!current.backendUrl && fallbackBackendUrl
              ? t("android.backendFallback", { url: fallbackBackendUrl })
              : t("android.backendHintPublic")}
          </p>
        </section>

        <section className="settings-section settings-advanced-section">
          <button
            type="button"
            className={`settings-accordion-trigger ${advancedOpen ? "is-open" : ""}`}
            onClick={() => setAdvancedOpen((value) => !value)}
            aria-expanded={advancedOpen}
          >
            <span className="settings-section-title is-inline">
              <span className="settings-section-icon" aria-hidden="true"><SlidersIcon /></span>
              <span>
                <strong>{t("android.advanced")}</strong>
                <small>{t("android.advancedDesc")}</small>
              </span>
            </span>
            <span aria-hidden="true">{advancedOpen ? "▾" : "▸"}</span>
          </button>

          {advancedOpen && (
            <div className="settings-advanced">
              <h4>{t("android.resultDisplay")}</h4>
              <ToggleRow
                label={t("android.showPredictionView")}
                help={t("android.showPredictionViewHelp")}
                checked={current.showPredictionView !== false}
                onChange={(value) => update({ showPredictionView: value })}
              />
              <ToggleRow
                label={t("android.showConfidence")}
                help={t("android.showConfidenceHelp")}
                checked={current.showConfidence !== false}
                onChange={(value) => update({ showConfidence: value })}
              />
              <label className="settings-field">
                <span>{t("android.breedsToShow")}</span>
                <select
                  className="form-input"
                  disabled={current.showPredictionView === false}
                  value={current.showPredictionView === false ? "0" : (current.topBreedsCount || 3)}
                  onChange={(e) => update({ topBreedsCount: Number(e.target.value) })}
                >
                  {current.showPredictionView === false && (
                    <option value="0">{t("android.mainResultOnly")}</option>
                  )}
                  {[1, 2, 3, 4, 5].map((count) => (
                    <option key={count} value={count}>{count}</option>
                  ))}
                </select>
              </label>
              <p className="settings-hint">{t("android.breedsToShowHelp")}</p>

              <h4>{t("android.oodGate")}</h4>
              <ToggleRow
                label={t("android.enableOod")}
                help={t("android.enableOodHelp")}
                checked={current.oodEnabled}
                onChange={(value) => update({ oodEnabled: value, activeOodProfile: value ? current.activeOodProfile : "disabled" })}
              />
              <label className="settings-field">
                <span>{t("android.oodProfile")}</span>
                <select
                  className="form-input"
                  value={current.activeOodProfile || "auto"}
                  onChange={(e) => selectOodProfile(e.target.value)}
                >
                  <option value="auto">{t("android.oodAuto")}</option>
                  <option value="disabled">{t("android.oodDisabled")}</option>
                  {oodProfiles.map((profile) => (
                    <option key={profile.filename} value={profile.filename}>{profile.filename}</option>
                  ))}
                </select>
              </label>
              {activeOod && (
                <p className="settings-hint">
                  {t("android.oodLoaded", {
                    file: activeOod.filename,
                    energy: activeOod.json_data.energy_threshold,
                    msp: activeOod.json_data.msp_threshold,
                  })}
                </p>
              )}
              {needsManualOod && (
                <div className="settings-warning">{t("android.oodWarning")}</div>
              )}
              <div className="settings-threshold-grid">
                <label className="settings-field">
                  <span>{t("android.energyThreshold")}</span>
                  <input
                    type="number" step="0.0001" className="form-input"
                    value={current.energyThreshold}
                    onChange={(e) => update({ energyThreshold: Number(e.target.value) })}
                  />
                </label>
                <label className="settings-field">
                  <span>{t("android.mspThreshold")}</span>
                  <input
                    type="number" step="0.0001" className="form-input"
                    value={current.mspThreshold}
                    onChange={(e) => update({ mspThreshold: Number(e.target.value) })}
                  />
                </label>
                <label className="settings-field">
                  <span>{t("android.binaryFloor")}</span>
                  <input
                    type="number" step="0.01" className="form-input"
                    value={current.binaryConfidenceFloor}
                    onChange={(e) => update({ binaryConfidenceFloor: Number(e.target.value) })}
                  />
                </label>
                <label className="settings-field">
                  <span>{t("android.temperature")}</span>
                  <input
                    type="number" step="0.01" className="form-input"
                    value={current.energyTemperature}
                    onChange={(e) => update({ energyTemperature: Number(e.target.value) })}
                  />
                </label>
              </div>

              <h4>{t("android.onDeviceModel")}</h4>
              <label className="settings-field">
                <span>{t("android.downscaleResolution")}</span>
                <select
                  className="form-input"
                  value={current.downscaleMode || "auto"}
                  onChange={(e) => update({ downscaleMode: e.target.value })}
                >
                  <option value="auto">{t("android.downscaleAuto")}</option>
                  <option value="train">{t("android.downscaleTrain")}</option>
                </select>
              </label>
              <p className="settings-hint">{t("android.downscaleHint")}</p>
              <label className="settings-field">
                <span>{t("android.activeModelLabel")}</span>
                <select
                  ref={modelSelectRef}
                  className="form-input"
                  value={current.activeModel}
                  onChange={(e) => selectModel(e.target.value)}
                >
                  {models.map((model) => (
                    <option
                      key={model.name}
                      value={model.name}
                      disabled={model.runnable === false}
                    >
                      {model.name} — {Number(model.size_mb || 0).toFixed(2)} MB
                      {model.runnable === false ? ` (${t("android.needsOnnx")})` : ""}
                    </option>
                  ))}
                </select>
              </label>
              <button type="button" className="btn btn-secondary btn-full" onClick={() => modelInputRef.current?.click()} disabled={busy}>
                {t("android.uploadModel")}
              </button>
              <input ref={modelInputRef} type="file" accept=".onnx,.pt" hidden onChange={handleModelUpload} />
              <p className="settings-hint">{t("android.modelHint")}</p>

              <h4>{t("android.oodProfiles")}</h4>
              <button type="button" className="btn btn-secondary btn-full" onClick={() => oodInputRef.current?.click()} disabled={busy}>
                {t("android.uploadOod")}
              </button>
              <input ref={oodInputRef} type="file" accept=".json,application/json" hidden onChange={handleOodUpload} />
              <div className="settings-ood-list">
                {oodProfiles.length === 0 && <p className="settings-hint">{t("android.noOodProfiles")}</p>}
                {oodProfiles.map((profile) => (
                  <div key={profile.filename} className="settings-ood-row">
                    <span>{profile.filename}</span>
                    <button
                      type="button"
                      className="settings-danger-link"
                      onClick={async () => {
                        await deleteOodProfile(profile.filename);
                        await onOodChanged?.();
                        if (current.activeOodProfile === profile.filename) {
                          await update({ activeOodProfile: "auto" });
                        }
                      }}
                    >
                      {t("android.remove")}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <footer className="settings-footer">
          <span>{t("android.modelsCount", { models: models.length, ood: oodProfiles.length })}</span>
          <button type="button" className="settings-save-btn" onClick={onClose}>
            ✓ {t("android.saveReturn")}
          </button>
        </footer>
      </aside>
    </div>
  );
}
