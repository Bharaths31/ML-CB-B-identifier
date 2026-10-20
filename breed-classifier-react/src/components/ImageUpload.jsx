import { useRef, useState } from "react";
import ImageCropper from "./ImageCropper.jsx";
import { useI18n } from "../i18n/I18nProvider.jsx";

/**
 * Minimal upload surface. It only shows the original image and the actions a
 * user needs. Optimizer fit modes, model-input preview, metadata and batch
 * testing live in the header Advanced panel.
 *
 * The cropper always opens `original.dataUrl`, so repeated framing is always
 * applied to the uploaded original — never to a previously cropped result.
 */
export default function ImageUpload({
  original,
  busy = false,
  error = null,
  framed = false,
  onFileSelected,
  onApplyCrop,
  onClear,
  onClassify,
  analyzing = false,
  canClassify = false,
  showClassify = true,
}) {
  const { t } = useI18n();
  const [isDragging, setIsDragging] = useState(false);
  const [editing, setEditing] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };
  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) processFile(file);
  };
  const handleFileInput = (event) => {
    const file = event.target.files?.[0];
    if (file) processFile(file);
    event.target.value = "";
  };

  const processFile = (file) => {
    if (!file.type.startsWith("image/")) {
      alert(t("upload.invalid"));
      return;
    }
    onFileSelected?.(file);
  };

  const handleConfirmCrop = (file) => {
    setEditing(false);
    onApplyCrop?.(file);
  };

  if (editing && original) {
    return (
      <div className="upload-container upload-editing">
        <ImageCropper
          imageUrl={original.dataUrl}
          fileName={original.name || "image.jpg"}
          onConfirm={handleConfirmCrop}
          onCancel={() => setEditing(false)}
          confirmLabel={t("optimizer.keep")}
        />
      </div>
    );
  }

  return (
    <div className="upload-container">
      {original ? (
        <div className="preview-card preview-card-square">
          <div className="preview-260-frame preview-original-frame">
            <img src={original.dataUrl} alt="Original uploaded livestock" />
            <span className="preview-260-badge">{t("optimizer.original")}</span>
            {busy && (
              <div className="optimizer-busy-overlay optimizer-busy-soft">
                <span className="camera-spinner" />
                <span>{t("optimizer.optimizing")}</span>
              </div>
            )}
          </div>

          {error && <p className="camera-error-inline">{error}</p>}

          <div className="preview-actions">
            <button
              type="button"
              className="btn btn-outline"
              onClick={() => setEditing(true)}
              disabled={busy}
            >
              {t("optimizer.adjust")}
            </button>
            <button type="button" className="btn btn-secondary" onClick={onClear} disabled={busy}>
              {t("actions.remove")}
            </button>
            <button
              type="button"
              className="btn btn-outline"
              onClick={() => fileInputRef.current?.click()}
              disabled={busy}
            >
              {t("actions.chooseAnother")}
            </button>
          </div>

          {showClassify && onClassify && (
            <div className="upload-classify-block">
              {framed && <p className="upload-classify-note">{t("upload.framedHint")}</p>}
              <button
                type="button"
                className="btn btn-primary btn-large btn-full classify-framed-btn"
                onClick={onClassify}
                disabled={!canClassify}
              >
                {analyzing ? (
                  <span className="btn-spinner-content">
                    <span className="spinner-border" />
                    {t("actions.analyzing")}
                  </span>
                ) : (
                  <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8" />
                      <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    </svg>
                    {framed ? t("actions.identifyFramed") : t("actions.identify")}
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      ) : (
        <div
          className={`dropzone ${isDragging ? "dropzone-active" : ""}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              fileInputRef.current?.click();
            }
          }}
        >
          <div className="dropzone-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
              <path d="M12 12v9" />
              <path d="m16 16-4-4-4 4" />
            </svg>
          </div>
          <div className="dropzone-text">
            <h3 className="dropzone-title">{t("upload.dropzoneTitle")}</h3>
            <p className="dropzone-subtitle">
              {t("upload.dropzoneSubtitle", { browse: "" }).trimEnd()}{" "}
              <span className="highlight-text">{t("upload.browse")}</span>
            </p>
            <p className="dropzone-hint">{t("upload.hint")}</p>
          </div>
        </div>
      )}

      {!original && error && <p className="camera-error-inline">{error}</p>}

      <input
        id="image-upload-input"
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={handleFileInput}
      />
    </div>
  );
}
