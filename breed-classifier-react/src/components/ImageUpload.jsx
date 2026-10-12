import { useState, useRef } from "react";

export default function ImageUpload({ onImageSelected, currentPreview, onClear }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

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
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      processFile(files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file) => {
    if (!file.type.startsWith("image/")) {
      alert("Please upload an image file (.jpg, .jpeg, .png, .webp).");
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      onImageSelected(file, ev.target.result);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="upload-container">
      {currentPreview ? (
        <div className="preview-card">
          <div className="preview-image-wrapper">
            <img src={currentPreview} alt="Selected livestock" className="preview-image" />
          </div>
          <div className="preview-actions">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClear}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
              Remove Image
            </button>
            <button
              type="button"
              className="btn btn-outline"
              onClick={() => fileInputRef.current?.click()}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              Choose Another
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
        >
          <div className="dropzone-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"></path>
              <path d="M12 12v9"></path>
              <path d="m16 16-4-4-4 4"></path>
            </svg>
          </div>
          <div className="dropzone-text">
            <h3 className="dropzone-title">Upload Livestock Image</h3>
            <p className="dropzone-subtitle">
              Drag & drop a photo here, or <span className="highlight-text">browse files</span>
            </p>
            <p className="dropzone-hint">Supports JPEG, PNG, WEBP high-resolution photographs</p>
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
