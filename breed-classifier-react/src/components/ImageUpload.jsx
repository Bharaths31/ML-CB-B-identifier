import { useEffect, useRef, useState } from "react";
import ImageCropper from "./ImageCropper.jsx";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { fileToDataUrl } from "../utils/cropAndQuality.js";

export default function ImageUpload({ onImageSelected, currentPreview, onClear }) {
  const { t } = useI18n();
  const [isDragging, setIsDragging] = useState(false);
  const [draft, setDraft] = useState(null); // { url, name }
  const [editing, setEditing] = useState(false);
  const fileInputRef = useRef(null);
  const originalRef = useRef(null);

  // A parent-provided preview (e.g. camera handoff) wins when not editing.
  useEffect(() => {
    if (currentPreview && !editing) {
      setDraft(null);
    }
  }, [currentPreview, editing]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  };
  const handleFileInput = (e) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    e.target.value = "";
  };

  const processFile = async (file) => {
    if (!file.type.startsWith("image/")) {
      alert(t("upload.invalid"));
      return;
    }
    try {
      const dataUrl = await fileToDataUrl(file);
      originalRef.current = { url: dataUrl, name: file.name };
      setDraft({ url: dataUrl, name: file.name });
      setEditing(true);
    } catch {
      alert(t("upload.invalid"));
    }
  };

  const handleConfirmCrop = (file, dataUrl, quality) => {
    setEditing(false);
    onImageSelected?.(file, dataUrl, quality);
  };

  const openAdjust = () => {
    const source = originalRef.current || (currentPreview ? { url: currentPreview, name: "uploaded.jpg" } : null);
    if (!source) return;
    setDraft(source);
    setEditing(true);
  };

  if (editing && draft) {
    return (
      <div className="upload-container upload-editing">
        <ImageCropper
          imageUrl={draft.url}
          fileName={draft.name}
          onConfirm={handleConfirmCrop}
          onCancel={() => setEditing(false)}
          confirmLabel={t("actions.useThis")}
        />
      </div>
    );
  }

  const preview = currentPreview;

  return (
    <div className="upload-container">
      {preview ? (
        <div className="preview-card preview-card-square">
          <div className="preview-image-wrapper preview-image-square">
            <img src={preview} alt="Selected livestock" className="preview-image" />
            <span className="preview-square-badge">1:1 · 260×260</span>
          </div>
          <div className="preview-actions">
            <button type="button" className="btn btn-outline" onClick={openAdjust}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 7V5a2 2 0 0 1 2-2h2" />
                <path d="M17 3h2a2 2 0 0 1 2 2v2" />
                <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
                <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
              </svg>
              {t("actions.adjust")}
            </button>
            <button type="button" className="btn btn-secondary" onClick={onClear}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
              {t("actions.remove")}
            </button>
            <button type="button" className="btn btn-outline" onClick={() => fileInputRef.current?.click()}>
              {t("actions.chooseAnother")}
            </button>
          </div>
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
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
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
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={handleFileInput}
      />
    </div>
  );
}
