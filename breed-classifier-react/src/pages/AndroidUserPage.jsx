import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { applyStoredTheme } from "../components/ThemeSwitcher.jsx";
import ImageCropper from "../components/ImageCropper.jsx";
import CameraCapture from "../components/CameraCapture.jsx";
import SettingsModal from "../components/SettingsModal.jsx";
import BreedTipsModal from "../components/BreedTipsModal.jsx";
import {
  AgeIcon,
  BrandMark,
  BookIcon,
  CameraIcon,
  FemaleIcon,
  GalleryIcon,
  LockIcon,
  MaleIcon,
  ShareIcon,
  TargetIcon,
  TrashIcon,
} from "../components/AndroidIcons.jsx";
import { useBackendHealth } from "../hooks/useBackendHealth.js";
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

const RECENT_KEY = "breedid_recent_scans";
const SCIENTIFIC_NAME = { Cattle: "Bos indicus", Buffalo: "Bubalus bubalis" };
const LANGUAGE_SHORT = { en: "EN", ta: "த", hi: "हि" };
const CROP_I18N = {
  pad: "android.cropPad",
  center: "android.cropCenter",
  manual: "android.cropManual",
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
  const { t, tBreed, tDynamic, lang } = useI18n();
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

  const backendStatus = useBackendHealth({
    enabled: settings.onlineMode,
    backendUrl: settings.backendUrl,
    online,
  });

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
  }, []);

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
      setError(err.message || t("android.prepareError"));
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
      setError(err.message || t("android.prepareError"));
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
      setError(err.message || t("android.prepareError"));
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
      setError(err.message || t("android.prepareError"));
    } finally {
      setPreparing(false);
    }
  };

  const analyze = async () => {
    if (!prepared?.canvas || !activeModel) {
      setError(t("android.chooseImageModel"));
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
      setError(err.message || t("android.inferenceError"));
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
      setToast(t("android.offlineTips"));
      return;
    }
    if (backendStatus !== "connected") {
      setToast(t("android.backendUnavailable"));
      return;
    }
    setTipsOpen(true);
  };

  const openSettings = () => {
    setSettingsOpen(true);
  };

  const shareCurrent = async () => {
    if (!result) return;
    const text = [
      `${tDynamic("species", result.species)} · ${tBreed(result.breed_slug) || result.breed}`,
      ...(settings.showConfidence !== false ? [`${t("result.confidence")}: ${result.confidence}%`] : []),
      `${t("hero.gender")}: ${tDynamic("gender", gender.label || "Unknown")}`,
      `${t("hero.age")}: ${tDynamic("age", age.key || age.label || "Unknown")}`,
    ].join("\n");
    try {
      if (navigator.share) {
        await navigator.share({ title: t("app.title"), text });
        return;
      }
    } catch {
      /* user cancelled or share unavailable */
    }
    const copied = await copyText(text);
    setToast(copied ? t("android.resultCopied") : t("android.shareUnavailable"));
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
  const showConfidence = settings.showConfidence !== false;
  const showPredictionView = settings.showPredictionView !== false;
  const topBreedsCount = Math.min(5, Math.max(1, Number(settings.topBreedsCount) || 3));
  const onlineModeActive = Boolean(settings.onlineMode && online);
  const headerConnected = onlineModeActive && backendStatus === "connected";
  const headerStatusLabel = !onlineModeActive
    ? t("android.offline")
    : backendStatus === "unconfigured"
      ? t("android.offline")
      : headerConnected
        ? t("header.connected")
        : t("header.reconnecting");
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
            aria-label={t("actions.back")}
          >
            ←
          </button>
          <div className="android-dossier-title">
            <h1>{t("android.dossier")}</h1>
            <span>{formatRecordId(result.scanned_at)}</span>
          </div>
          <button
            type="button"
            className="android-menu-btn"
            onClick={openSettings}
            aria-label={t("android.appSettings")}
          >
            ☰
          </button>
        </header>
      ) : (
        <header className="site-header android-header">
          <div className="brand-zone android-brand">
            <div className="brand-emblem android-brand-mark" aria-hidden="true">
              <BrandMark className="android-brand-icon" />
            </div>
            <div className="brand-text android-brand-text">
              <h1 className="brand-title android-brand-name">BreedID</h1>
              <span className="brand-subtitle android-brand-sub">Livestock Vision</span>
            </div>
          </div>
          <div className="android-header-actions">
            <span className={`android-online-pill ${headerConnected ? "is-online" : "is-offline"}`}>
              <span className={`telemetry-dot ${headerConnected ? "online" : "offline"}`} />
              {headerStatusLabel}
            </span>
            <button
              type="button"
              className="android-lang-chip"
              onClick={openSettings}
              aria-label={t("android.selectLanguage")}
              title={t("android.selectLanguage")}
            >
              <span>{LANGUAGE_SHORT[lang] || "EN"}</span>
              <span className="android-lang-glyph" aria-hidden="true">文</span>
            </button>
            <button
              type="button"
              className="android-menu-btn"
              onClick={openSettings}
              aria-label={t("mobile.menu")}
              title={t("android.appSettings")}
            >
              ☰
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
                  {t("android.visionActive")}
                </span>
                <span className="android-engine-tag">ONNX · WASM</span>
              </div>
              <h2>{t("android.identifyTitle")}</h2>
              <p>{t("android.identifyBody")}</p>
            </section>

            <div className="android-source-stack">
              <button
                type="button"
                className="android-source-btn"
                onClick={() => uploadRef.current?.click()}
                disabled={preparing}
              >
                <span className="android-source-icon" aria-hidden="true"><GalleryIcon /></span>
                <span className="android-source-copy">
                  <strong>{preparing ? t("android.preparing") : t("android.uploadGallery")}</strong>
                  <small>
                    {preparing ? t("android.preparingHint") : t("android.uploadHint")}
                  </small>
                </span>
                <span className="android-source-arrow" aria-hidden="true">›</span>
              </button>

              <button
                type="button"
                className="android-source-btn is-primary"
                onClick={() => setCameraOpen(true)}
                disabled={preparing}
              >
                <span className="android-source-icon" aria-hidden="true"><CameraIcon /></span>
                <span className="android-source-copy">
                  <strong>
                    {t("android.takePhoto")}
                    <em className="android-fast-badge">{t("android.fast")}</em>
                  </strong>
                  <small>{t("android.cameraHint")}</small>
                </span>
                <span className="android-source-arrow" aria-hidden="true">›</span>
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
                <span>{t("android.activeModel")}</span>
                <span className="android-telemetry-ready"><i aria-hidden="true" />WASM / SIMD</span>
              </div>
              <div className="android-model-name">{activeModel?.name || t("android.noModel")}</div>
              <div className="android-telemetry-bottom">
                <span>
                  {t("android.size")}: {activeModel ? Number(activeModel.size_mb || 0).toFixed(1) : "—"} MB
                  {" · "}
                  {t("android.latency")}: {modelLatency ? `${Math.round(modelLatency)}ms` : "—"}
                </span>
                <strong>{activeModel ? t("android.engineReady") : t("android.modelMissing")}</strong>
              </div>
            </section>

            <section className="android-recent-block">
              <div className="android-recent-head">
                <div className="android-recent-title">
                  <span>{t("android.recentScans")}</span>
                  <b className="android-count-badge">{recent.length}</b>
                </div>
                {recent.length > 0 && (
                  <button type="button" className="android-clear-btn" onClick={clearRecent}>
                    <TrashIcon />
                    {t("android.clear")}
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
                          : <span className="android-recent-placeholder" aria-hidden="true"><BrandMark /></span>}
                        {showConfidence && (
                          <b className={entry.confidence < 90 ? "is-warn" : ""}>
                            {entry.confidence}%
                          </b>
                        )}
                      </span>
                      <strong>{tBreed(entry.breed_slug) || entry.breed}</strong>
                      <small>{SCIENTIFIC_NAME[entry.species] || entry.species}</small>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="android-empty-recents">
                  <span aria-hidden="true"><AgeIcon /></span>
                  <strong>{t("android.noRecent")}</strong>
                  <small>{t("android.noRecentHint")}</small>
                </div>
              )}
            </section>

            <div className="android-privacy-strip">
              <span><LockIcon /> {t("android.localInference")}</span>
              <span>{t("android.noCloudSync")}</span>
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
                aria-label={t("actions.back")}
              >
                ←
              </button>
              <div className="android-preview-title">
                <span className="android-step-kicker">{t("android.imagePreparation")}</span>
                <h2>{source === "camera" ? t("android.cameraPreview") : t("android.preparePhoto")}</h2>
                <p>
                  {source === "camera" ? t("android.cameraBody") : t("android.prepareBody")}
                </p>
              </div>
            </div>

            <div className="android-preview-grid">
              <figure className="android-preview-frame">
                <span className="android-frame-tag">{t("android.original")}</span>
                <img src={original?.dataUrl} alt={t("android.original")} />
              </figure>
              <figure className="android-preview-frame">
                <span className="android-frame-tag is-accent">{t("android.modelInput")}</span>
                <img src={prepared?.dataUrl} alt={t("android.modelInput")} />
              </figure>
            </div>

            {source === "file" && (
              <div className="android-option-block">
                <span className="android-option-label">{t("android.cropMode")}</span>
                <div className="android-option-row">
                  {CROP_MODES.map((mode) => (
                    <button
                      key={mode.id}
                      type="button"
                      className={`android-chip ${cropMode === mode.id ? "is-active" : ""}`}
                      onClick={() => changeCropMode(mode.id)}
                      disabled={preparing}
                    >
                      {t(CROP_I18N[mode.id] || mode.id)}
                    </button>
                  ))}
                </div>
                <p className="android-option-hint">
                  {t("android.downscale")}: {(settings.downscaleMode || "auto") === "train"
                    ? t("android.downscaleTrain")
                    : t("android.downscaleAuto")} · {t("android.changeInSettings")}
                </p>
              </div>
            )}

            {manualOpen && original && (
              <div className="android-manual-crop">
                <ImageCropper
                  imageUrl={original.dataUrl}
                  fileName={original.file?.name || "upload.jpg"}
                  confirmLabel={t("android.useCrop")}
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
                  {t("android.analyzing")}
                </>
              ) : t("android.analyze")}
            </button>
          </section>
        )}

        {stage === "result" && result && (
          <section className="android-results android-dossier">
            <div className="android-inference-strip">
              <span className="android-inference-state">
                <i aria-hidden="true" />
                {t("android.inferenceComplete")}
              </span>
              <span>
                {resultLatency ? `${Math.round(resultLatency)}ms` : "on-device"}
                {" · "}{t("android.onDeviceNeural")}
              </span>
            </div>

            {resultImage && (
              <figure className="android-dossier-image">
                <img src={resultImage} alt={t("android.dossier")} />
                <figcaption>◉ {t("android.modelInput")}</figcaption>
              </figure>
            )}

            <article className="android-species-card">
              <div className="android-species-copy">
                <span className="android-card-label">{t("android.species")}</span>
                <h2>{speciesLabel}</h2>
                <small>{SCIENTIFIC_NAME[result.species] || result.species}</small>
              </div>
              <span className="android-verified-badge">◉ {t("android.modelMatch")}</span>
            </article>

            <article className="android-breed-dossier">
              <div className="android-dossier-breed-top">
                <span className="android-card-label">{t("android.identifiedBreed")}</span>
                <span className="android-rank-pill">{t("android.rank1")}</span>
              </div>
              <div className="android-breed-title-row">
                <h2>{breedLabel}</h2>
                {showConfidence && <strong>{result.confidence}%</strong>}
              </div>
              <span className="android-breed-subtitle">
                {speciesLabel} · {t("android.breedMatch")}
              </span>
              <p className="android-breed-desc">
                {showPredictionView ? t("android.breedDesc") : t("android.breedDescMain")}
              </p>

              {showPredictionView && (
                <div className="android-spectrum">
                  <div className="android-spectrum-head">
                    <span>{t("android.topBreeds", { count: topBreedsCount })}</span>
                    <span>{t("android.softmax")}</span>
                  </div>
                  {topBars(result.top5_breeds)
                    .filter((item) => item.species === result.species)
                    .slice(0, topBreedsCount)
                    .map((item, index) => (
                    <div className="android-spectrum-row" key={item.breed_slug}>
                      <div className="android-spectrum-line">
                        <span className="android-spectrum-name">
                          {index + 1}. {tBreed(item.breed_slug) || item.breed}
                        </span>
                        {showConfidence && <strong>{item.confidence}%</strong>}
                      </div>
                      <span className="android-spectrum-track">
                        <i style={{ width: `${item.width}%` }} />
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </article>

            <div className="android-attr-grid">
              <article className="android-attr-card">
                <div className="android-attr-head">
                  <span className="android-attr-icon" aria-hidden="true">
                    {gender.key === "female" ? <FemaleIcon /> : <MaleIcon />}
                  </span>
                  <span className="android-card-label">{t("android.predictedGender")}</span>
                </div>
                <h3>{genderLabel}</h3>
                {showConfidence && (
                  <strong className="android-attr-confidence">
                    {gender.confidence}% {t("android.heuristic")}
                  </strong>
                )}
                <small>{t("android.methodGender")}</small>
              </article>

              <article className="android-attr-card">
                <div className="android-attr-head">
                  <span className="android-attr-icon" aria-hidden="true"><AgeIcon /></span>
                  <span className="android-card-label">{t("android.estimatedAge")}</span>
                </div>
                <h3>{ageLabel}</h3>
                {showConfidence && (
                  <strong className="android-attr-confidence">
                    {age.confidence}% {t("android.heuristic")}
                  </strong>
                )}
                <small>{t("android.methodAge")}</small>
              </article>
            </div>

            <div className="android-action-stack">
              <button type="button" className="android-primary-btn" onClick={openTips}>
                <BookIcon /> {t("android.breedTips")}
              </button>
              <button type="button" className="android-secondary-btn" onClick={shareCurrent}>
                <ShareIcon /> {t("android.shareDossier")}
              </button>
              <button type="button" className="android-ghost-btn" onClick={startOver}>
                <TargetIcon /> {t("android.identifyAnother")}
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
        backendStatus={backendStatus}
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
