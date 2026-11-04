import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { formatBytes } from "../utils/imageOptimizer.js";
import AdvancedBatchTester from "./AdvancedBatchTester.jsx";
import AdvancedResultDetails from "./AdvancedResultDetails.jsx";

const FIT_OPTIONS = [
  { fitMode: "stretch", padStyle: "imagenet_mean", label: "optimizer.stretch" },
  { fitMode: "pad", padStyle: "imagenet_mean", label: "optimizer.padMean" },
  { fitMode: "pad", padStyle: "ambient_blur", label: "optimizer.padBlur" },
  { fitMode: "pad", padStyle: "black", label: "optimizer.padBlack" },
];

/**
 * Header-level Advanced modal dialog with subnav tabs (matching Developer Settings).
 * Exposes model-input preview/optimizer, vision diagnostics, and batch testing.
 */
export default function AdvancedPanel({
  open,
  onClose,
  optimizer,
  model,
  result,
  config,
  quality,
  previewUrl,
  modelInputUrl,
  onEscalate,
  onOpenExpert,
}) {
  const { t } = useI18n();
  const [view, setView] = useState("model");
  const [activeTab, setActiveTab] = useState(result ? "diagnostics" : "optimizer");

  useEffect(() => {
    if (result && open) {
      setActiveTab("diagnostics");
    } else if (!result && open) {
      setActiveTab("optimizer");
    }
  }, [open, Boolean(result)]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const meta = optimizer.result?.meta || null;
  const originalUrl = optimizer.original?.dataUrl || null;
  const modelUrl = optimizer.result?.dataUrl || originalUrl;
  const previewSrc = view === "original" ? originalUrl : modelUrl;

  const isActive = (option) =>
    optimizer.fit.fitMode === option.fitMode &&
    (option.fitMode === "stretch" || optimizer.fit.padStyle === option.padStyle);

  return (
    <div className="advanced-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <section className="advanced-panel" onClick={(event) => event.stopPropagation()}>
        <header className="advanced-head">
          <div>
            <h3>{t("advanced.title")}</h3>
            <p>{t("advanced.subtitle")}</p>
          </div>
          <button
            type="button"
            className="camera-close-icon"
            onClick={onClose}
            aria-label={t("actions.close")}
          >
            ✕
          </button>
        </header>

        {/* Subnavigation Bar (styled like Developer Settings dev-subnav-bar) */}
        <nav className="advanced-subnav-bar" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "optimizer"}
            className={`advanced-nav-btn ${activeTab === "optimizer" ? "advanced-nav-active" : ""}`}
            onClick={() => setActiveTab("optimizer")}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
            <span>{t("advanced.tabOptimizer")}</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "diagnostics"}
            className={`advanced-nav-btn ${activeTab === "diagnostics" ? "advanced-nav-active" : ""}`}
            onClick={() => setActiveTab("diagnostics")}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
            <span>{t("advanced.tabDiagnostics")}</span>
            {result && <span className="advanced-tab-badge" title="Diagnostics ready" />}
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "batch"}
            className={`advanced-nav-btn ${activeTab === "batch" ? "advanced-nav-active" : ""}`}
            onClick={() => setActiveTab("batch")}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
            <span>{t("advanced.tabBatch")}</span>
          </button>
        </nav>

        <div className="advanced-body">
          {activeTab === "optimizer" && (
            <>
              {!originalUrl ? (
                <div className="advanced-empty">{t("advanced.noImage")}</div>
              ) : (
                <>
                  <div className="advanced-preview-block">
                    <div className="preview-260-frame">
                      <img src={previewSrc} alt="Advanced optimizer preview" />
                      <span className="preview-260-badge">
                        {view === "original" ? t("optimizer.original") : "260×260"}
                      </span>
                      {optimizer.busy && (
                        <div className="optimizer-busy-overlay">
                          <span className="camera-spinner" />
                          <span>{t("optimizer.optimizing")}</span>
                        </div>
                      )}
                    </div>

                    <div className="advanced-preview-side">
                      <div className="optimizer-preview-toggle advanced-view-toggle">
                        <button
                          type="button"
                          className={`optimizer-toggle-btn ${view === "model" ? "is-on" : ""}`}
                          onClick={() => setView("model")}
                          disabled={!optimizer.result}
                        >
                          {t("optimizer.viewModel")}
                        </button>
                        <button
                          type="button"
                          className={`optimizer-toggle-btn ${view === "original" ? "is-on" : ""}`}
                          onClick={() => setView("original")}
                          disabled={!originalUrl}
                        >
                          {t("optimizer.viewOriginal")}
                        </button>
                      </div>

                      <div className="advanced-fit-controls" role="group" aria-label={t("optimizer.fitMode")}>
                        {FIT_OPTIONS.map((option) => (
                          <button
                            key={`${option.fitMode}-${option.padStyle}`}
                            type="button"
                            className={`optimizer-mode-btn ${isActive(option) ? "is-on" : ""}`}
                            onClick={() =>
                              optimizer.setFitMode({
                                fitMode: option.fitMode,
                                padStyle: option.padStyle,
                              })
                            }
                            disabled={optimizer.busy}
                          >
                            {t(option.label)}
                          </button>
                        ))}
                      </div>

                      <div className="control-setting-row clahe-row">
                        <div className="setting-info">
                          <span className="setting-label">{t("camera.clahe")}</span>
                          <span className="setting-subtext">{t("camera.claheHint")}</span>
                        </div>
                        <label className="toggle-switch">
                          <input
                            type="checkbox"
                            checked={Boolean(optimizer.clahe)}
                            onChange={(event) => optimizer.setClaheEnabled?.(event.target.checked)}
                            disabled={optimizer.busy}
                          />
                          <span className="toggle-slider" />
                        </label>
                      </div>
                      {optimizer.error && <p className="camera-error-inline">{optimizer.error}</p>}
                      {optimizer.original?.name && (
                        <p className="advanced-file-name">
                          {t("optimizer.file")}: <strong>{optimizer.original.name}</strong>
                        </p>
                      )}
                    </div>
                  </div>

                  {meta && (
                    <div className="advanced-meta-grid">
                      <div>
                        <span>{t("optimizer.original")}</span>
                        <strong>{meta.originalWidth}×{meta.originalHeight}</strong>
                      </div>
                      <div>
                        <span>{t("optimizer.modelInput")}</span>
                        <strong>260×260</strong>
                      </div>
                      <div>
                        <span>{t("optimizer.scaled")}</span>
                        <strong>{meta.scaledInnerWidth}×{meta.scaledInnerHeight}</strong>
                      </div>
                      <div>
                        <span>{t("optimizer.padding")}</span>
                        <strong>{meta.padX}, {meta.padY}</strong>
                      </div>
                      <div>
                        <span>{t("optimizer.bytes")}</span>
                        <strong>{formatBytes(meta.optimizedBytes)}</strong>
                      </div>
                      <div>
                        <span>{t("optimizer.time")}</span>
                        <strong>{meta.durationMs} ms</strong>
                      </div>
                    </div>
                  )}
                </>
              )}
            </>
          )}

          {activeTab === "diagnostics" && (
            <>
              {result ? (
                <AdvancedResultDetails
                  result={result}
                  config={config}
                  quality={quality}
                  previewUrl={previewUrl || originalUrl}
                  modelInputUrl={modelInputUrl || modelUrl}
                  optimizerMeta={meta}
                  onEscalate={onEscalate}
                  onOpenExpert={onOpenExpert}
                />
              ) : (
                <div className="advanced-empty">
                  {t("advanced.noDiagnostics")}
                </div>
              )}
            </>
          )}

          {activeTab === "batch" && (
            <AdvancedBatchTester fit={optimizer.fit} model={model} />
          )}
        </div>
      </section>
    </div>
  );
}
