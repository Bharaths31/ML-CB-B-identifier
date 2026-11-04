import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { applyStoredTheme } from "../components/ThemeSwitcher.jsx";
import ImageCropper from "../components/ImageCropper.jsx";
import CameraCapture from "../components/CameraCapture.jsx";
import SettingsModal from "../components/SettingsModal.jsx";
import BreedTipsModal from "../components/BreedTipsModal.jsx";
import { AGE_OPTIONS, GENDER_OPTIONS } from "../services/breedCatalog.js";
import { CROP_MODES, prepareImage } from "../services/imagePrep.js";
import {
  DEFAULT_SETTINGS,
  ensureDefaultModel,
  getSettings,
  listModels,
  listOodProfiles,
  runOnDeviceInference,
  saveSettings,
  fetchBreedKnowledgeOnline,
} from "../services/onDeviceInference.js";

const OFFLINE_TIPS_MESSAGE =
  "📡 Offline Mode Active — Turn ON 'Online Mode' in Settings and connect to the internet to load detailed Breed Tips & Farmer Advisory data.";
const RECENT_KEY = "breedid_recent_scans";

function haptic(pattern = 20) {
  try { navigator.vibrate?.(pattern); } catch { /* ignore */ }
}

function readRecent() {
  try {
    const parsed = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
    return Array.isArray(parsed) ? parsed.slice(0, 5) : [];
  } catch {
    return [];
  }
}

function writeRecent(entries) {
  try { localStorage.setItem(RECENT_KEY, JSON.stringify(entries.slice(0, 5))); } catch { /* ignore */ }
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

function topBars(top5 = []) {
  const max = Math.max(...top5.map((item) => item.confidence || 0), 1);
  return top5.map((item) => ({ ...item, width: Math.max(4, (item.confidence / max) * 100) }));
}

export default function AndroidUserPage() {
  const { t, tBreed, tDynamic } = useI18n();
  const uploadRef = useRef(null);

  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [models, setModels] = useState([]);
  const [oodProfiles, setOodProfiles] = useState([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [tipsOpen, setTipsOpen] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [recent, setRecent] = useState(readRecent);
  const [online, setOnline] = useState(() => (typeof navigator === "undefined" ? true : navigator.onLine));
  const wakeLockRef = useRef(null);

  const [stage, setStage] = useState("input"); // input | preview | result
  const [source, setSource] = useState("file"); // file | camera
  const [original, setOriginal] = useState(null);
  const [prepared, setPrepared] = useState(null);
  const [cropMode, setCropMode] = useState("pad");
  const [manualOpen, setManualOpen] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [toast, setToast] = useState("");
  const [genderOverride, setGenderOverride] = useState(null);
  const [ageOverride, setAgeOverride] = useState(null);

  const refreshModels = useCallback(async () => {
    const rows = await listModels();
    setModels(rows);
    return rows;
  }, []);

  const refreshOod = useCallback(async () => {
    const rows = await listOodProfiles();
    setOodProfiles(rows);
    return rows;
  }, []);

  useEffect(() => {
    applyStoredTheme();
    (async () => {
      await ensureDefaultModel().catch(() => null);
      const [loadedSettings, loadedModels] = await Promise.all([getSettings(), refreshModels()]);
      setSettings(loadedSettings);
      await refreshOod();
      if (!loadedModels.some((m) => m.name === loadedSettings.activeModel) && loadedModels[0]) {
        const next = await saveSettings({ activeModel: loadedModels[0].name });
        setSettings(next);
      }
    })();
  }, [refreshModels, refreshOod]);

  // Development/debug hook used by the automated browser smoke test.
  useEffect(() => {
    const debug = typeof window !== "undefined"
      && new URLSearchParams(window.location.search).get("debug") === "1";
    if (import.meta.env.DEV || debug) {
      window.__breedTest = {
        prepareImage,
        runOnDeviceInference,
        ensureDefaultModel,
        listModels,
        getSettings,
        saveSettings,
        fetchBreedKnowledgeOnline,
      };
    }
  }, []);

  useEffect(() => {
    const update = () => setOnline(navigator.onLine);
    window.addEventListener("online", update);
    window.addEventListener("offline", update);
    return () => {
      window.removeEventListener("online", update);
      window.removeEventListener("offline", update);
    };
  }, []);

  // Keep the screen awake while the camera or inference is active.
  useEffect(() => {
    let cancelled = false;
    const acquire = async () => {
      if (!(cameraOpen || analyzing) || !navigator.wakeLock?.request) return;
      try {
        const lock = await navigator.wakeLock.request("screen");
        if (!cancelled) wakeLockRef.current = lock;
      } catch { /* ignored: some WebViews deny wake lock */ }
    };
    acquire();
    return () => {
      cancelled = true;
      try { wakeLockRef.current?.release?.(); } catch { /* ignore */ }
      wakeLockRef.current = null;
    };
  }, [cameraOpen, analyzing]);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = setTimeout(() => setToast(""), 6500);
    return () => clearTimeout(timer);
  }, [toast]);

  const activeModel = useMemo(
    () => models.find((model) => model.name === settings.activeModel) || models[0] || null,
    [models, settings.activeModel],
  );
  const activeOodProfile = useMemo(
    () => oodProfiles.find((profile) => profile.filename === settings.activeOodProfile) || null,
    [oodProfiles, settings.activeOodProfile],
  );

  const applySettings = async (patch) => {
    const next = await saveSettings(patch);
    setSettings(next);
    return next;
  };

  const resetResult = () => {
    setResult(null);
    setGenderOverride(null);
    setAgeOverride(null);
    setToast("");
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setError("");
    setPreparing(true);
    resetResult();
    setSource("file");
    try {
      const loaded = await prepareImage(file, {
        cropMode,
        scaleMode: settings.downscaleMode || "auto",
      });
      setOriginal({ file, dataUrl: loaded.dataUrl, width: loaded.meta.originalWidth, height: loaded.meta.originalHeight });
      // Re-read the original data URL from the source file for the preview.
      const raw = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.readAsDataURL(file);
      });
      setOriginal({ file, dataUrl: raw, width: loaded.meta.originalWidth, height: loaded.meta.originalHeight });
      setPrepared(loaded);
      setStage("preview");
    } catch (err) {
      setError(err.message || "Could not prepare the image.");
    } finally {
      setPreparing(false);
    }
  };

  const handleCameraCapture = async (file, dataUrl, meta, report) => {
    setCameraOpen(false);
    setError("");
    setPreparing(true);
    resetResult();
    setSource("camera");
    try {
      const prepared = await prepareImage(file, { cropMode: "center", scaleMode: "auto" });
      setOriginal({
        file,
        dataUrl: dataUrl || prepared.dataUrl,
        width: meta?.originalWidth || 260,
        height: meta?.originalHeight || 260,
      });
      setPrepared(prepared);
      setStage("preview");
      haptic(25);
    } catch (err) {
      setError(err.message || "Could not prepare the camera image.");
    } finally {
      setPreparing(false);
    }
  };

  const rePrepare = async (nextCrop = cropMode) => {
    if (!original?.file || source === "camera") return;
    setPreparing(true);
    setError("");
    try {
      const loaded = await prepareImage(original.file, {
        cropMode: nextCrop,
        scaleMode: settings.downscaleMode || "auto",
      });
      setPrepared(loaded);
    } catch (err) {
      setError(err.message || "Could not prepare the image.");
    } finally {
      setPreparing(false);
    }
  };

  const changeCropMode = async (value) => {
    setCropMode(value);
    if (value === "manual") {
      setManualOpen(true);
      return;
    }
    await rePrepare(value);
  };

  const applyManualCrop = async (file) => {
    setManualOpen(false);
    if (!original?.file) return;
    setPreparing(true);
    try {
      const loaded = await prepareImage(original.file, {
        cropMode: "center",
        scaleMode: settings.downscaleMode || "auto",
        manualCropFile: file,
      });
      setPrepared(loaded);
      setCropMode("manual");
    } catch (err) {
      setError(err.message || "Could not apply the crop.");
    } finally {
      setPreparing(false);
    }
  };

  const analyze = async () => {
    if (!prepared?.canvas || !activeModel) {
      setError("Choose an image and an ONNX model first.");
      return;
    }
    setAnalyzing(true);
    setError("");
    try {
      const res = await runOnDeviceInference({
        canvas: prepared.canvas,
        model: activeModel,
        settings,
        oodProfile: activeOodProfile,
        imageMeta: prepared.meta,
      });
      setResult(res);
      setGenderOverride(null);
      setAgeOverride(null);
      setStage("result");
      haptic([15, 30, 15]);
      const entry = {
        ...res,
        thumbnail: prepared.dataUrl,
        scanned_at: new Date().toISOString(),
      };
      setRecent((previous) => {
        const next = [entry, ...previous].slice(0, 5);
        writeRecent(next);
        return next;
      });
    } catch (err) {
      setError(err.message || "On-device inference failed.");
    } finally {
      setAnalyzing(false);
    }
  };

  const startOver = () => {
    resetResult();
    setOriginal(null);
    setPrepared(null);
    setCropMode("pad");
    setSource("file");
    setStage("input");
  };

  const openTips = () => {
    if (!settings.onlineMode || !navigator.onLine) {
      setToast(OFFLINE_TIPS_MESSAGE);
      return;
    }
    setTipsOpen(true);
  };

  const shareCurrent = async () => {
    if (!result) return;
    const text = [
      `${result.species} · ${tBreed(result.breed_slug) || result.breed}`,
      `Confidence: ${result.confidence}%`,
      `Gender: ${selectedGenderLabel}`,
      `Age: ${selectedAgeLabel}`,
    ].join("\n");
    try {
      if (navigator.share) {
        await navigator.share({ title: "Breed Classifier", text });
        return;
      }
    } catch {
      /* user cancelled or share unavailable */
    }
    const copied = await copyText(text);
    setToast(copied ? "Result copied to clipboard." : "Sharing is unavailable on this device.");
  };

  const openRecent = (entry) => {
    setResult(entry);
    setGenderOverride(null);
    setAgeOverride(null);
    setStage("result");
    haptic(12);
  };

  const clearRecent = () => {
    writeRecent([]);
    setRecent([]);
  };

  const gender = result?.gender || { key: null, label: "Unknown", confidence: 0 };
  const age = result?.age_group || { key: null, label: "Unknown", confidence: 0 };
  const selectedGender = genderOverride || gender.key;
  const selectedAge = ageOverride || age.key;
  const selectedGenderLabel = GENDER_OPTIONS.find((o) => o.key === selectedGender)?.label || gender.label;
  const selectedAgeLabel = AGE_OPTIONS.find((o) => o.key === selectedAge)?.label || age.label;

  return (
    <div className="android-shell">
      <header className="site-header android-header">
        <div className="brand-zone">
          <div className="brand-emblem" aria-hidden="true">🐄</div>
          <div className="brand-text">
            <h1 className="brand-title">{t("app.title")}</h1>
            <span className="brand-subtitle">On-device breed identification</span>
          </div>
        </div>
        <div className="android-header-actions">
          <span className={`settings-online-badge ${online ? "is-online" : "is-offline"}`}>
            <span className={`telemetry-dot ${online ? "online" : "offline"}`} />
            {online ? "Online" : "Offline"}
          </span>
          <button
            type="button"
            className="android-menu-btn"
            onClick={() => setSettingsOpen(true)}
            aria-label="Open settings menu"
            title="Settings"
          >
            ☰
          </button>
        </div>
      </header>

      <main className="android-main">
        {stage === "input" && (
          <section className="android-card android-input-card">
            <h2>Identify a Cattle or Buffalo Breed</h2>
            <p>All inference runs on this phone with ONNX Runtime Web. No internet is required.</p>
            <div className="android-source-grid">
              <button type="button" className="android-source-btn" onClick={() => uploadRef.current?.click()}>
                <span aria-hidden="true">🖼️</span>
                <strong>Upload from Gallery</strong>
                <small>Choose crop & downscale before analysis</small>
              </button>
              <button type="button" className="android-source-btn" onClick={() => setCameraOpen(true)}>
                <span aria-hidden="true">📷</span>
                <strong>Take Photo</strong>
                <small>Guided silhouette + auto framing</small>
              </button>
            </div>
            <input
              ref={uploadRef}
              type="file"
              accept=".png,.jpg,.jpeg,.bmp,.webp,image/*"
              hidden
              onChange={handleUpload}
            />
            {error && <div className="error-banner">{error}</div>}
            <div className="android-model-note">
              Active model: <strong>{activeModel?.name || "none"}</strong>
              {activeModel && <span> · {Number(activeModel.size_mb || 0).toFixed(1)} MB</span>}
            </div>

            {recent.length > 0 && (
              <div className="android-recent">
                <div className="android-recent-head">
                  <span>Recent scans</span>
                  <button type="button" className="settings-danger-link" onClick={clearRecent}>Clear</button>
                </div>
                <div className="android-recent-grid">
                  {recent.map((entry) => (
                    <button
                      key={`${entry.scanned_at}-${entry.breed_slug}`}
                      type="button"
                      className="android-recent-card"
                      onClick={() => openRecent(entry)}
                    >
                      {entry.thumbnail && <img src={entry.thumbnail} alt="" />}
                      <span>{tBreed(entry.breed_slug) || entry.breed}</span>
                      <small>{entry.confidence}% · {entry.species}</small>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}

        {stage === "preview" && (
          <section className="android-card">
            <div className="android-preview-head">
              <div>
                <h2>{source === "camera" ? "Camera Preview" : "Prepare Your Photo"}</h2>
                <p>
                  {source === "camera"
                    ? "Camera captures use the default Resize(260) + CenterCrop(260) pipeline."
                    : "Choose how the uploaded photo should be fitted before analysis."}
                </p>
              </div>
              <button type="button" className="btn btn-outline" onClick={startOver}>Cancel</button>
            </div>

            <div className="android-preview-grid">
              <figure className="android-preview-frame">
                <img src={original?.dataUrl} alt="Original upload" />
                <figcaption>Original</figcaption>
              </figure>
              <figure className="android-preview-frame">
                <img src={prepared?.dataUrl} alt="Prepared model input" />
                <figcaption>Model input · 260×260</figcaption>
              </figure>
            </div>

            {source === "file" && (
              <>
                <div className="android-option-block">
                  <span className="android-option-label">Crop Mode</span>
                  <div className="android-option-row">
                    {CROP_MODES.map((mode) => (
                      <button
                        key={mode.id}
                        type="button"
                        className={`preset-chip ${cropMode === mode.id ? "active" : ""}`}
                        onClick={() => changeCropMode(mode.id)}
                        disabled={preparing}
                      >
                        {mode.label}
                      </button>
                    ))}
                  </div>
                  <p className="android-option-hint">
                    Downscale: {(settings.downscaleMode || "auto") === "train"
                      ? "Train Match (288→260)"
                      : "Auto (shortest side 260)"} · change in Settings
                  </p>
                </div>
              </>
            )}

            {manualOpen && original && (
              <div className="android-manual-crop">
                <ImageCropper
                  imageUrl={original.dataUrl}
                  fileName={original.file?.name || "upload.jpg"}
                  confirmLabel="Use this crop"
                  onConfirm={applyManualCrop}
                  onCancel={() => setManualOpen(false)}
                />
              </div>
            )}

            {error && <div className="error-banner">{error}</div>}
            <button type="button" className="btn btn-primary btn-large btn-full" onClick={analyze} disabled={analyzing || preparing}>
              {analyzing ? "Analyzing on device…" : "Analyze Breed"}
            </button>
          </section>
        )}

        {stage === "result" && result && (
          <section className="android-results">
            <div className="android-result-grid">
              <article className="android-result-card">
                <span className="android-card-label">Species</span>
                <h3>{tDynamic("species", result.species)}</h3>
                <span className="android-confidence">{result.species_confidence}%</span>
              </article>

              <article className="android-result-card android-breed-card">
                <span className="android-card-label">Breed</span>
                <h3>{tBreed(result.breed_slug) || result.breed}</h3>
                <span className="android-confidence">{result.confidence}%</span>
                <div className="android-top5">
                  {topBars(result.top5_breeds).slice(0, 3).map((item) => (
                    <div key={item.breed_slug} className="android-top5-row">
                      <span>{tBreed(item.breed_slug) || item.breed}</span>
                      <div className="android-top5-track">
                        <span style={{ width: `${item.width}%` }} />
                      </div>
                      <strong>{item.confidence}%</strong>
                    </div>
                  ))}
                </div>
              </article>

              <article className="android-result-card">
                <span className="android-card-label">Gender</span>
                <h3>{selectedGenderLabel}</h3>
                <span className="android-confidence">{gender.confidence}%</span>
                <div className="android-pill-row">
                  {GENDER_OPTIONS.map((option) => (
                    <button
                      key={option.key}
                      type="button"
                      className={`hero-pill ${selectedGender === option.key ? "hero-pill-active" : ""}`}
                      onClick={() => setGenderOverride(option.key)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </article>

              <article className="android-result-card">
                <span className="android-card-label">Age Group</span>
                <h3>{selectedAgeLabel}</h3>
                <span className="android-confidence">{age.confidence}%</span>
                <div className="android-pill-row">
                  {AGE_OPTIONS.map((option) => (
                    <button
                      key={option.key}
                      type="button"
                      className={`hero-pill ${selectedAge === option.key ? "hero-pill-active" : ""}`}
                      onClick={() => setAgeOverride(option.key)}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </article>
            </div>

            <div className="android-result-actions">
              <button type="button" className="btn btn-primary btn-large" onClick={openTips}>
                📖 Breed Tips & Farmer Advisory
              </button>
              <button type="button" className="btn btn-secondary" onClick={shareCurrent}>
                Share result
              </button>
            </div>

            <button type="button" className="btn btn-outline btn-full" onClick={startOver}>
              Identify another animal
            </button>
          </section>
        )}
      </main>

      {toast && <div className="android-toast" role="status">{toast}</div>}

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        models={models}
        oodProfiles={oodProfiles}
        onSettingsChange={applySettings}
        onModelsChanged={refreshModels}
        onOodChanged={refreshOod}
      />

      <BreedTipsModal
        open={tipsOpen}
        onClose={() => setTipsOpen(false)}
        settings={settings}
        result={result}
        gender={selectedGenderLabel}
        ageGroup={selectedAgeLabel}
      />

      {cameraOpen && (
        <CameraCapture
          androidMode
          onCapture={handleCameraCapture}
          onClose={() => setCameraOpen(false)}
        />
      )}
    </div>
  );
}
