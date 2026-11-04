import { useEffect, useRef, useState } from "react";
import { THEMES } from "./ThemeSwitcher.jsx";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { LANGUAGES } from "../i18n/LanguageContext.jsx";
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
  onSettingsChange,
  onModelsChanged,
  onOodChanged,
}) {
  const { lang, setLang } = useI18n();
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [online, setOnline] = useState(() => (typeof navigator === "undefined" ? true : navigator.onLine));
  const modelInputRef = useRef(null);
  const oodInputRef = useRef(null);
  const modelSelectRef = useRef(null);

  const current = settings || DEFAULT_SETTINGS;
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
          setMessage(`Added ${record.name}. Embedded OOD calibration detected and applied.`);
        } else {
          await update({
            activeModel: record.name,
            activeOodProfile: "auto",
            oodEnabled: false,
          });
          setMessage(`Added ${record.name}. No embedded OOD calibration — upload an OOD .json below to enable the OOD gate.`);
        }
      } else {
        setMessage(`Saved ${record.name}. .pt checkpoints are listed below but must be exported to ONNX before on-device inference.`);
      }
      requestAnimationFrame(() => {
        modelSelectRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
        modelSelectRef.current?.focus();
      });
    } catch (err) {
      setMessage(err.message || "Model upload failed.");
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
      setMessage(`${name}: embedded OOD calibration applied.`);
    } else {
      await update({ activeModel: name, activeOodProfile: "auto", oodEnabled: false });
      setMessage(`${name}: no embedded OOD calibration. Upload an OOD .json to enable the OOD gate.`);
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
      setMessage(`OOD profile ${record.filename} saved.`);
    } catch (err) {
      setMessage(err.message || "OOD JSON upload failed.");
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
        aria-label="Settings"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="settings-head">
          <button
            type="button"
            className="settings-back"
            onClick={onClose}
            aria-label="Close settings"
          >
            ←
          </button>
          <div className="settings-head-copy">
            <h2>App Settings</h2>
            <p>On-device model, language and appearance</p>
          </div>
          <span className="settings-status-badge">
            <i aria-hidden="true" />
            Offline ready
          </span>
        </header>

        {message && <div className="settings-message">{message}</div>}

        <section className="settings-section settings-language-section">
          <div className="settings-section-title">
            <span className="settings-section-icon" aria-hidden="true">文</span>
            <div>
              <h3>Select Language</h3>
              <p>Applies to buttons, results and breed names.</p>
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
            <span className="settings-section-icon" aria-hidden="true">◐</span>
            <div>
              <h3>Theme Display</h3>
              <p>Optimised for bright outdoor daylight and night barns.</p>
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
                    <small>{theme.desc}</small>
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
            <span className="settings-section-icon" aria-hidden="true">🌐</span>
            <div>
              <h3>Online Mode</h3>
              <p>Only Breed Tips need the network. Inference stays on-device.</p>
            </div>
          </div>
          <ToggleRow
            label="Online Mode"
            help="Fetch Breed Tips & Farmer Advisory from the backend. On-device inference always works offline."
            checked={current.onlineMode}
            onChange={(value) => update({ onlineMode: value })}
          />
          <div className={`settings-online-badge ${online ? "is-online" : "is-offline"}`}>
            <span className={`telemetry-dot ${online ? "online" : "offline"}`} />
            {online ? "Online" : "Offline"}
            <span className="settings-online-extra">
              {current.onlineMode ? "Online mode ON" : "Online mode OFF"}
            </span>
          </div>
          <label className="settings-field">
            <span>Backend URL (tips only)</span>
            <input
              type="url"
              className="form-input"
              value={current.backendUrl || ""}
              placeholder="http://192.168.1.20:5000"
              onChange={(e) => update({ backendUrl: e.target.value })}
            />
          </label>
          <p className="settings-hint">
            Android emulator: <code>http://10.0.2.2:5000</code>. Physical phone: use the
            computer's LAN IP and keep the Flask server running.
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
              <span className="settings-section-icon" aria-hidden="true">⚙</span>
              <span>
                <strong>Advanced AI Model</strong>
                <small>ONNX engine, OOD safety and thresholds</small>
              </span>
            </span>
            <span aria-hidden="true">{advancedOpen ? "▾" : "▸"}</span>
          </button>

          {advancedOpen && (
            <div className="settings-advanced">
              <h4>Out-of-Distribution Gate</h4>
              <ToggleRow
                label="Enable OOD"
                help="Reject images outside the model's breed knowledge."
                checked={current.oodEnabled}
                onChange={(value) => update({ oodEnabled: value, activeOodProfile: value ? current.activeOodProfile : "disabled" })}
              />
              <label className="settings-field">
                <span>OOD Profile</span>
                <select
                  className="form-input"
                  value={current.activeOodProfile || "auto"}
                  onChange={(e) => selectOodProfile(e.target.value)}
                >
                  <option value="auto">Auto (embedded / defaults)</option>
                  <option value="disabled">Disabled</option>
                  {oodProfiles.map((profile) => (
                    <option key={profile.filename} value={profile.filename}>{profile.filename}</option>
                  ))}
                </select>
              </label>
              {activeOod && (
                <p className="settings-hint">
                  Loaded {activeOod.filename}: energy {activeOod.json_data.energy_threshold}, MSP {activeOod.json_data.msp_threshold}.
                </p>
              )}
              {needsManualOod && (
                <div className="settings-warning">
                  This model has no embedded OOD calibration. Upload an OOD .json below or turn OOD off.
                </div>
              )}
              <div className="settings-threshold-grid">
                <label className="settings-field">
                  <span>Energy threshold</span>
                  <input
                    type="number" step="0.0001" className="form-input"
                    value={current.energyThreshold}
                    onChange={(e) => update({ energyThreshold: Number(e.target.value) })}
                  />
                </label>
                <label className="settings-field">
                  <span>MSP threshold</span>
                  <input
                    type="number" step="0.0001" className="form-input"
                    value={current.mspThreshold}
                    onChange={(e) => update({ mspThreshold: Number(e.target.value) })}
                  />
                </label>
                <label className="settings-field">
                  <span>Binary floor</span>
                  <input
                    type="number" step="0.01" className="form-input"
                    value={current.binaryConfidenceFloor}
                    onChange={(e) => update({ binaryConfidenceFloor: Number(e.target.value) })}
                  />
                </label>
                <label className="settings-field">
                  <span>Temperature</span>
                  <input
                    type="number" step="0.01" className="form-input"
                    value={current.energyTemperature}
                    onChange={(e) => update({ energyTemperature: Number(e.target.value) })}
                  />
                </label>
              </div>

              <h4>On-Device Model</h4>
              <label className="settings-field">
                <span>Downscale resolution (uploaded photos)</span>
                <select
                  className="form-input"
                  value={current.downscaleMode || "auto"}
                  onChange={(e) => update({ downscaleMode: e.target.value })}
                >
                  <option value="auto">Auto (shortest side 260px)</option>
                  <option value="train">Train Match (288px → 260px)</option>
                </select>
              </label>
              <p className="settings-hint">
                Applies to gallery uploads. Camera captures always use the default pipeline.
              </p>
              <label className="settings-field">
                <span>Active model</span>
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
                      {model.runnable === false ? " (needs ONNX export)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <button type="button" className="btn btn-secondary btn-full" onClick={() => modelInputRef.current?.click()} disabled={busy}>
                + Upload .onnx / .pt
              </button>
              <input ref={modelInputRef} type="file" accept=".onnx,.pt" hidden onChange={handleModelUpload} />
              <p className="settings-hint">
                On-device inference runs <code>.onnx</code>. A <code>.pt</code> checkpoint must
                be exported to ONNX from the desktop developer console first.
              </p>

              <h4>OOD Profiles</h4>
              <button type="button" className="btn btn-secondary btn-full" onClick={() => oodInputRef.current?.click()} disabled={busy}>
                + Upload OOD .json
              </button>
              <input ref={oodInputRef} type="file" accept=".json,application/json" hidden onChange={handleOodUpload} />
              <div className="settings-ood-list">
                {oodProfiles.length === 0 && <p className="settings-hint">No custom profiles yet.</p>}
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
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <footer className="settings-footer">
          <span>{models.length} model(s) · {oodProfiles.length} OOD profile(s)</span>
          <button type="button" className="settings-save-btn" onClick={onClose}>
            ✓ Save & Return
          </button>
        </footer>
      </aside>
    </div>
  );
}
