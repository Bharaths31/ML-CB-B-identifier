import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { applyStoredTheme } from "../components/ThemeSwitcher.jsx";
import ImageCropper from "../components/ImageCropper.jsx";
import CameraCapture from "../components/CameraCapture.jsx";
import SettingsModal from "../components/SettingsModal.jsx";
import BreedTipsModal from "../components/BreedTipsModal.jsx";
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
  "ðŸ“¡ Offline Mode Active â€” Turn ON 'Online Mode' in Settings and connect to the internet to load detailed Breed Tips & Farmer Advisory data.";
const RECENT_KEY = "breedid_recent_scans";
const SCIENTIFIC_NAME = { Cattle: "Bos indicus", Buffalo: "Bubalus bubalis" };
const LANGUAGE_SHORT = { en: "EN", ta: "à®¤", hi: "à¤¹à¤¿" };
const HELP_TEXT = {
  en: "Upload a saved photo or take a new one. Keep the whole animal side profile inside the frame, then tap Analyze Breed.",
  ta: "à®šà¯‡à®®à®¿à®¤à¯à®¤ à®ªà¯à®•à¯ˆà®ªà¯à®ªà®Ÿà®¤à¯à®¤à¯ˆ à®ªà®¤à®¿à®µà¯‡à®±à¯à®±à®µà¯à®®à¯ à®…à®²à¯à®²à®¤à¯ à®ªà¯à®¤à®¿à®¯ à®ªà®Ÿà®®à¯ à®Žà®Ÿà¯à®•à¯à®•à®µà¯à®®à¯. à®µà®¿à®²à®™à¯à®•à®¿à®©à¯ à®®à¯à®´à¯ à®ªà®•à¯à®•à®¤à¯ à®¤à¯‹à®±à¯à®±à®¤à¯à®¤à¯ˆ à®šà®Ÿà¯à®Ÿà®•à®¤à¯à®¤à®¿à®²à¯ à®µà¯ˆà®¤à¯à®¤à¯, à®ªà®¿à®©à¯à®©à®°à¯ à®‡à®©à®¤à¯à®¤à¯ˆ à®ªà®•à¯à®ªà¯à®ªà®¾à®¯à¯à®µà¯ à®šà¯†à®¯à¯à®¯à®µà¯à®®à¯.",
  hi: "à¤¸à¤¹à¥‡à¤œà¥€ à¤—à¤ˆ à¤«à¥‹à¤Ÿà¥‹ à¤…à¤ªà¤²à¥‹à¤¡ à¤•à¤°à¥‡à¤‚ à¤¯à¤¾ à¤¨à¤ˆ à¤«à¥‹à¤Ÿà¥‹ à¤²à¥‡à¤‚à¥¤ à¤œà¤¾à¤¨à¤µà¤° à¤•à¤¾ à¤ªà¥‚à¤°à¤¾ à¤¸à¤¾à¤‡à¤¡ à¤ªà¥à¤°à¥‹à¤«à¤¾à¤‡à¤² à¤«à¥à¤°à¥‡à¤® à¤®à¥‡à¤‚ à¤°à¤–à¥‡à¤‚, à¤«à¤¿à¤° à¤¨à¤¸à¥à¤² à¤•à¤¾ à¤µà¤¿à¤¶à¥à¤²à¥‡à¤·à¤£ à¤•à¤°à¥‡à¤‚à¥¤",
};

function formatRecordId(value) {
  const date = value ? new Date(value) : new Date();
  const pad = (number) => String(number).padStart(2, "0");
  return `REC-${date.getFullYear()}-${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}`;
}

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
  return top5.map((item) => ({
    ...item,
    width: Math.max(3, Math.min(100, item.confidence || 0)),
  }));
}

export default function AndroidUserPage() {
  const { tBreed, tDynamic, lang, speakText, stopSpeaking } = useI18n();
  const uploadRef = useRef(null);
  const originalUrlRef = useRef(null);

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
  const [lastLatency, setLastLatency] = useState(null);
  const [speaking, setSpeaking] = useState(false);

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

  useEffect(() => {
    if (typeof history !== "undefined" && "scrollRestoration" in history) {
      history.scrollRestoration = "manual";
    }
    window.scrollTo(0, 0);
  }, []);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [stage]);

  // Release the original-preview object URL when the page goes away.
  useEffect(() => () => {
    if (originalUrlRef.current) URL.revokeObjectURL(originalUrlRef.current);
    originalUrlRef.current = null;
    stopSpeaking();
  }, [stopSpeaking]);

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
    setToast("");
  };

  const releaseOriginalUrl = () => {
    if (!originalUrlRef.current) return;
    URL.revokeObjectURL(originalUrlRef.current);
    originalUrlRef.current = null;
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
      // Object URLs avoid base64-encoding the full-resolution file just for
      // the preview; the image is already decoded once inside prepareImage.
      releaseOriginalUrl();
      const previewUrl = URL.createObjectURL(file);
      originalUrlRef.current = previewUrl;
      setOriginal({
        file,
        dataUrl: previewUrl,
        width: loaded.meta.originalWidth,
        height: loaded.meta.originalHeight,
      });
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
      releaseOriginalUrl();
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
      const scannedAt = new Date().toISOString();
      setResult({ ...res, scanned_at: scannedAt });
      setLastLatency(res.inference_time_ms ?? res.total_time_ms ?? null);
      setStage("result");
      haptic([15, 30, 15]);
      const entry = {
        ...res,
        scanned_at: scannedAt,
        thumbnail: prepared.dataUrl,
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
    releaseOriginalUrl();
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

  const openSettings = () => {
    stopSpeaking();
    setSpeaking(false);
    setSettingsOpen(true);
  };

  const toggleHelp = () => {
    if (speaking) {
      stopSpeaking();
      setSpeaking(false);
      return;
    }
    const started = speakText(HELP_TEXT[lang] || HELP_TEXT.en, {
      onEnd: () => setSpeaking(false),
    });
    setSpeaking(started);
    if (!started) setToast("Audio help is not available on this device.");
  };

  const shareCurrent = async () => {
    if (!result) return;
    const text = [
      `${result.species} Â· ${tBreed(result.breed_slug) || result.breed}`,
      `Confidence: ${result.confidence}%`,
      `Gender: ${genderLabel}`,
      `Age: ${ageLabel}`,
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
    setStage("result");
    haptic(12);
  };

  const clearRecent = () => {
    writeRecent([]);
    setRecent([]);
  };

  const gender = result?.gender || { key: null, label: "Unknown", confidence: 0 };
  const age = result?.age_group || { key: null, label: "Unknown", confidence: 0 };
  const resultImage = result?.thumbnail || prepared?.dataUrl || original?.dataUrl || null;
  const resultLatency = result?.inference_time_ms ?? result?.total_time_ms ?? lastLatency;
  const modelLatency = lastLatency ?? recent[0]?.inference_time_ms ?? null;
  const speciesLabel = tDynamic("species", result?.species || "");
  const breedLabel = tBreed(result?.breed_slug) || result?.breed || "";
  const genderLabel = tDynamic("gender", gender.label || "Unknown");
  const ageLabel = tDynamic("age", age.key || age.label || "Unknown");

  return (
    <div className="android-shell">
      {stage === "result" && result ? (
        <header className="site-header android-header android-dossier-header">
          <button
            type="button"
            className="android-back-btn"
            onClick={startOver}
            aria-label="Back to identify"
          >
            â†
          </button>
          <div className="android-dossier-title">
            <h1>Identification Dossier</h1>
            <span>{formatRecordId(result.scanned_at)}</span>
          </div>
          <button
            type="button"
            className="android-menu-btn"
            onClick={openSettings}
            aria-label="Open settings menu"
          >
            â˜°
          </button>
        </header>
      ) : (
        <header className="site-header android-header">
          <div className="brand-zone android-brand">
            <div className="brand-emblem android-brand-mark" aria-hidden="true">ðŸ„</div>
            <div className="brand-text android-brand-text">
              <h1 className="brand-title android-brand-name">BreedID</h1>
              <span className="brand-subtitle android-brand-sub">Livestock Vision</span>
            </div>
          </div>
          <div className="android-header-actions">
            <span className={`android-online-pill ${online ? "is-online" : "is-offline"}`}>
              <span className={`telemetry-dot ${online ? "online" : "offline"}`} />
              {online ? "Online" : "Offline"}
            </span>
            <button
              type="button"
              className="android-lang-chip"
              onClick={openSettings}
              aria-label="Open language settings"
              title="Language settings"
            >
              <span>{LANGUAGE_SHORT[lang] || "EN"}</span>
              <span className="android-lang-glyph" aria-hidden="true">æ–‡</span>
            </button>
            <button
              type="button"
              className="android-menu-btn"
              onClick={openSettings}
              aria-label="Open settings menu"
              title="Settings"
            >
              â˜°
            </button>
          </div>
        </header>
      )}

      <main className="android-main">
        {stage === "input" && (
          <>
            <section className="android-card android-hero-card">
              <div className="android-engine-row">
                <span className="android-engine-state">
                  <i aria-hidden="true" />
                  Vision Engine Active
                </span>
                <span className="android-engine-tag">ONNX Â· WASM</span>
              </div>
              <h2>Identify Cattle or Buffalo</h2>
              <p>
                Surgical on-device computer vision. Works offline in harsh field conditions
                with zero cloud latency.
              </p>
            </section>

            <button
              type="button"
              className={`android-audio-help ${speaking ? "is-speaking" : ""}`}
              onClick={toggleHelp}
              aria-pressed={speaking}
            >
              <span className="android-audio-icon" aria-hidden="true">{speaking ? "â– " : "ðŸ”Š"}</span>
              <span className="android-audio-copy">
                <strong>{lang === "ta" ? "à®•à¯‡à®Ÿà¯à®Ÿà¯ à®…à®±à®¿à®¯à¯à®™à¯à®•à®³à¯" : lang === "hi" ? "à¤¸à¥à¤¨à¤•à¤° à¤œà¤¾à¤¨à¥‡à¤‚" : "Audio help"}</strong>
                <small>{speaking ? "Playing instructionsâ€¦" : "Tap to hear how to scan in your language"}</small>
              </span>
              <span className="android-audio-play" aria-hidden="true">{speaking ? "â– " : "â–¶"}</span>
            </button>

            <div className="android-source-stack">
              <button
                type="button"
                className="android-source-btn"
                onClick={() => uploadRef.current?.click()}
                disabled={preparing}
              >
                <span className="android-source-icon" aria-hidden="true">ðŸ–¼ï¸</span>
                <span className="android-source-copy">
                  <strong>{preparing ? "Preparing imageâ€¦" : "Upload from Gallery"}</strong>
                  <small>
                    {preparing
                      ? "Optimising for the model"
                      : "Saved photos Â· choose crop & downscale"}
                  </small>
                </span>
                <span className="android-source-arrow" aria-hidden="true">â€º</span>
              </button>

              <button
                type="button"
                className="android-source-btn is-primary"
                onClick={() => setCameraOpen(true)}
                disabled={preparing}
              >
                <span className="android-source-icon" aria-hidden="true">ðŸ“·</span>
                <span className="android-source-copy">
                  <strong>
                    Take Photo
                    <em className="android-fast-badge">FAST</em>
                  </strong>
                  <small>Guided silhouette Â· direct on-device scan</small>
                </span>
                <span className="android-source-arrow" aria-hidden="true">â€º</span>
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

            <section className="android-telemetry-card">
              <div className="android-telemetry-top">
                <span>Active ONNX model</span>
                <span className="android-telemetry-ready"><i aria-hidden="true" />WASM / SIMD</span>
              </div>
              <div className="android-model-name">{activeModel?.name || "No model loaded"}</div>
              <div className="android-telemetry-bottom">
                <span>
                  Size: {activeModel ? Number(activeModel.size_mb || 0).toFixed(1) : "â€”"} MB
                  {" Â· "}
                  Latency: {modelLatency ? `${Math.round(modelLatency)}ms` : "â€”"}
                </span>
                <strong>{activeModel ? "Engine Ready" : "Model Missing"}</strong>
              </div>
            </section>

            <section className="android-recent-block">
              <div className="android-recent-head">
                <div className="android-recent-title">
                  <span>Recent Scans</span>
                  <b className="android-count-badge">{recent.length}</b>
                </div>
                {recent.length > 0 && (
                  <button type="button" className="android-clear-btn" onClick={clearRecent}>
                    ðŸ—‘ Clear
                  </button>
                )}
              </div>
              {recent.length > 0 ? (
                <div className="android-recent-grid">
                  {recent.map((entry) => (
                    <button
                      key={`${entry.scanned_at}-${entry.breed_slug}`}
                      type="button"
                      className="android-recent-card"
                      onClick={() => openRecent(entry)}
                    >
                      <span className="android-recent-thumb">
                        {entry.thumbnail
                          ? <img src={entry.thumbnail} alt="" />
                          : <span className="android-recent-placeholder" aria-hidden="true">ðŸ„</span>}
                        <b className={entry.confidence < 90 ? "is-warn" : ""}>
                          {entry.confidence}%
                        </b>
                      </span>
                      <strong>{tBreed(entry.breed_slug) || entry.breed}</strong>
                      <small>{SCIENTIFIC_NAME[entry.species] || entry.species}</small>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="android-empty-recents">
                  <span aria-hidden="true">â—·</span>
                  <strong>No Recent Local Scans</strong>
                  <small>Processed samples will appear here in the offline cache.</small>
                </div>
              )}
            </section>

            <div className="android-privacy-strip">
              <span>ðŸ”’ Local in-memory inference</span>
              <span>No cloud sync</span>
            </div>
          </>
        )}

        {stage === "preview" && (
          <section className="android-card android-prepare-card">
            <div className="android-preview-head">
              <button
                type="button"
                className="android-back-btn is-compact"
                onClick={startOver}
                aria-label="Back to home"
              >
                â†
              </button>
              <div className="android-preview-title">
                <span className="android-step-kicker">Image preparation</span>
                <h2>{source === "camera" ? "Camera Preview" : "Prepare Your Photo"}</h2>
                <p>
                  {source === "camera"
                    ? "Camera captures use the default Resize(260) + CenterCrop(260) pipeline."
                    : "Choose how the uploaded photo should be fitted before analysis."}
                </p>
              </div>
            </div>

            <div className="android-preview-grid">
              <figure className="android-preview-frame">
                <span className="android-frame-tag">Original</span>
                <img src={original?.dataUrl} alt="Original upload" />
              </figure>
              <figure className="android-preview-frame">
                <span className="android-frame-tag is-accent">Model input Â· 260Ã—260</span>
                <img src={prepared?.dataUrl} alt="Prepared model input" />
              </figure>
            </div>

            {source === "file" && (
              <div className="android-option-block">
                <span className="android-option-label">Crop Mode</span>
                <div className="android-option-row">
                  {CROP_MODES.map((mode) => (
                    <button
                      key={mode.id}
                      type="button"
                      className={`android-chip ${cropMode === mode.id ? "is-active" : ""}`}
                      onClick={() => changeCropMode(mode.id)}
                      disabled={preparing}
                    >
                      {mode.label}
                    </button>
                  ))}
                </div>
                <p className="android-option-hint">
                  Downscale: {(settings.downscaleMode || "auto") === "train"
                    ? "Train Match (288â†’260)"
                    : "Auto (shortest side 260)"} Â· change in Settings
                </p>
              </div>
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
            <button
              type="button"
              className="android-primary-btn"
              onClick={analyze}
              disabled={analyzing || preparing}
            >
              {analyzing ? (
                <>
                  <span className="android-btn-spinner" aria-hidden="true" />
                  Analyzing on deviceâ€¦
                </>
              ) : "Analyze Breed"}
            </button>
          </section>
        )}

        {stage === "result" && result && (
          <section className="android-results android-dossier">
            <div className="android-inference-strip">
              <span className="android-inference-state">
                <i aria-hidden="true" />
                Inference complete
              </span>
              <span>
                {resultLatency ? `${Math.round(resultLatency)}ms` : "on-device"}
                {" Â· "}On-device neural
              </span>
            </div>

            {resultImage && (
              <figure className="android-dossier-image">
                <img src={resultImage} alt="Analysed animal" />
                <figcaption>â—‰ Model input Â· 260Ã—260</figcaption>
              </figure>
            )}

            <article className="android-species-card">
              <div className="android-species-copy">
                <span className="android-card-label">Species</span>
                <h2>{speciesLabel}</h2>
                <small>{SCIENTIFIC_NAME[result.species] || result.species}</small>
              </div>
              <span className="android-verified-badge">â—‰ Model match</span>
            </article>

            <article className="android-breed-dossier">
              <div className="android-dossier-breed-top">
                <span className="android-card-label">Identified breed</span>
                <span className="android-rank-pill">Rank #1 probable</span>
              </div>
              <div className="android-breed-title-row">
                <h2>{breedLabel}</h2>
                <strong>{result.confidence}%</strong>
              </div>
              <span className="android-breed-subtitle">
                {speciesLabel} Â· on-device breed match
              </span>
              <p className="android-breed-desc">
                Ranked from 68 cattle and buffalo breeds in the offline model. The top
                three softmax probabilities are shown below.
              </p>

              <div className="android-spectrum">
                <div className="android-spectrum-head">
                  <span>Top-3 confidence spectrum</span>
                  <span>Softmax distribution</span>
                </div>
                {topBars(result.top5_breeds)
                  .filter((item) => item.species === result.species)
                  .slice(0, 3)
                  .map((item, index) => (
                  <div className="android-spectrum-row" key={item.breed_slug}>
                    <div className="android-spectrum-line">
                      <span className="android-spectrum-name">
                        {index + 1}. {tBreed(item.breed_slug) || item.breed}
                      </span>
                      <strong>{item.confidence}%</strong>
                    </div>
                    <span className="android-spectrum-track">
                      <i style={{ width: `${item.width}%` }} />
                    </span>
                  </div>
                ))}
              </div>
            </article>

            <div className="android-attr-grid">
              <article className="android-attr-card">
                <div className="android-attr-head">
                  <span className="android-attr-icon" aria-hidden="true">
                    {gender.key === "female" ? "â™€" : "â™‚"}
                  </span>
                  <span className="android-card-label">Predicted gender</span>
                </div>
                <h3>{genderLabel}</h3>
                <strong className="android-attr-confidence">{gender.confidence}% heuristic</strong>
                <small>{gender.method || "On-device morphology prior"}</small>
              </article>

              <article className="android-attr-card">
                <div className="android-attr-head">
                  <span className="android-attr-icon" aria-hidden="true">â—·</span>
                  <span className="android-card-label">Estimated age</span>
                </div>
                <h3>{ageLabel}</h3>
                <strong className="android-attr-confidence">{age.confidence}% heuristic</strong>
                <small>{age.method || "On-device body-size / coat prior"}</small>
              </article>
            </div>

            <div className="android-action-stack">
              <button type="button" className="android-primary-btn" onClick={openTips}>
                ðŸ“– Breed Tips & Farmer Advisory
              </button>
              <button type="button" className="android-secondary-btn" onClick={shareCurrent}>
                â†— Share Dossier Export
              </button>
              <button type="button" className="android-ghost-btn" onClick={startOver}>
                â—Ž Identify Another Animal
              </button>
            </div>
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
        gender={genderLabel}
        ageGroup={ageLabel}
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
