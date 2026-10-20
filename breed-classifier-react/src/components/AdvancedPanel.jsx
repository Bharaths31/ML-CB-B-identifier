import { useEffect, useState } from "react";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { formatBytes } from "../utils/imageOptimizer.js";
import AdvancedBatchTester from "./AdvancedBatchTester.jsx";

const FIT_OPTIONS = [
  { fitMode: "stretch", padStyle: "imagenet_mean", label: "optimizer.stretch" },
  { fitMode: "pad", padStyle: "imagenet_mean", label: "optimizer.padMean" },
  { fitMode: "pad", padStyle: "ambient_blur", label: "optimizer.padBlur" },
  { fitMode: "pad", padStyle: "black", label: "optimizer.padBlack" },
];

/**
 * Header-level Advanced panel. Keeps the normal page minimal while exposing
 * the model-input preview, optimizer metadata, fit modes and batch testing.
 */
export default function AdvancedPanel({ open, onClose, optimizer, model }) {
  const { t } = useI18n();
  const [view, setView] = useState("model");

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
          <button type="button" className="camera-close-icon" onClick={onClose} aria-label={t("actions.close")}>
            ✕
          </button>
        </header>

        <div className="advanced-body">
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

          <AdvancedBatchTester fit={optimizer.fit} model={model} />
        </div>
      </section>
    </div>
  );
}
