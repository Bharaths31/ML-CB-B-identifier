import { useEffect, useRef, useState } from "react";

/**
 * Modern Live Camera Capture modal with:
 * - Real-time viewfinder with targeting frame & live indicator
 * - Front/rear camera toggle (facingMode: user vs environment)
 * - Flash capture animation
 * - Freeze-frame review (Retake vs Accept)
 */
export default function CameraCapture({ onCapture, onClose }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState(null);
  const [facingMode, setFacingMode] = useState("environment");
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [capturedBlob, setCapturedBlob] = useState(null);
  const [capturedPreview, setCapturedPreview] = useState(null);
  const [isFlashing, setIsFlashing] = useState(false);
  const [isStarting, setIsStarting] = useState(true);

  // Check if multiple cameras are available
  useEffect(() => {
    async function checkDevices() {
      try {
        if (!navigator.mediaDevices?.enumerateDevices) return;
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter((d) => d.kind === "videoinput");
        setHasMultipleCameras(videoInputs.length > 1);
      } catch (err) {
        console.warn("Could not enumerate video devices:", err);
      }
    }
    checkDevices();
  }, []);

  // Start / restart camera stream whenever facingMode changes
  useEffect(() => {
    let cancelled = false;
    setIsStarting(true);
    setError(null);

    async function startCamera() {
      // Stop any existing stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }

      try {
        const constraints = {
          video: {
            facingMode: { ideal: facingMode },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        };

        let stream;
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (exactErr) {
          // Fallback to any available video if exact constraint fails
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
        setIsStarting(false);
      } catch (err) {
        if (!cancelled) {
          setError(
            err.name === "NotAllowedError" || err.name === "PermissionDeniedError"
              ? "Camera permission denied. Please allow camera access in your browser settings."
              : err.name === "NotFoundError"
              ? "No camera device found on this system."
              : `Unable to access camera: ${err.message || "Unknown error"}`
          );
          setIsStarting(false);
        }
      }
    }

    startCamera();

    return () => {
      cancelled = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, [facingMode]);

  // Clean up snapshot preview blob url
  useEffect(() => {
    return () => {
      if (capturedPreview) {
        URL.revokeObjectURL(capturedPreview);
      }
    };
  }, [capturedPreview]);

  // Handle escape key
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const triggerCapture = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;

    // Trigger visual shutter flash
    setIsFlashing(true);
    setTimeout(() => setIsFlashing(false), 200);

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");

    // If using user (front) facing camera, mirror image for natural photo
    if (facingMode === "user") {
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        setCapturedBlob(blob);
        setCapturedPreview(URL.createObjectURL(blob));
      },
      "image/jpeg",
      0.95
    );
  };

  const retake = () => {
    if (capturedPreview) URL.revokeObjectURL(capturedPreview);
    setCapturedBlob(null);
    setCapturedPreview(null);
  };

  const confirmCapture = () => {
    if (capturedBlob) {
      onCapture(capturedBlob);
      onClose();
    }
  };

  const switchCamera = () => {
    setFacingMode((prev) => (prev === "environment" ? "user" : "environment"));
  };

  return (
    <div className="camera-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="camera-modal modern-camera-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="camera-modal-header">
          <div className="camera-title-wrap">
            <span className="camera-pulse-dot" />
            <h3>{capturedPreview ? "Review Photo" : "Live Camera"}</h3>
          </div>
          <button className="camera-close-icon" onClick={onClose} title="Close camera">
            ✕
          </button>
        </div>

        {/* Viewport */}
        <div className="camera-viewport-wrap">
          {isFlashing && <div className="camera-flash" />}

          {error ? (
            <div className="camera-error-container">
              <span className="camera-error-icon">📷</span>
              <p className="camera-error-msg">{error}</p>
              <button
                className="btn ghost btn-sm"
                onClick={() => setFacingMode((f) => (f === "environment" ? "user" : "environment"))}
              >
                🔄 Retry Camera
              </button>
            </div>
          ) : capturedPreview ? (
            <div className="camera-preview-frozen">
              <img src={capturedPreview} alt="Captured livestock" />
              <div className="camera-frozen-badge">✓ Photo Captured</div>
            </div>
          ) : (
            <div className="camera-stream-container">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={facingMode === "user" ? "mirrored" : ""}
              />
              {isStarting && (
                <div className="camera-loading-overlay">
                  <div className="camera-spinner" />
                  <span>Accessing camera stream…</span>
                </div>
              )}
              {/* Viewfinder Overlay */}
              <div className="camera-reticle">
                <div className="corner top-left" />
                <div className="corner top-right" />
                <div className="corner bottom-left" />
                <div className="corner bottom-right" />
                <div className="reticle-guide-text">Center animal in frame</div>
              </div>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="camera-controls-footer">
          {capturedPreview ? (
            <div className="camera-review-actions">
              <button className="btn ghost" onClick={retake}>
                🔄 Retake Photo
              </button>
              <button className="btn primary" onClick={confirmCapture}>
                ✓ Use This Photo
              </button>
            </div>
          ) : (
            <div className="camera-live-actions">
              {hasMultipleCameras && (
                <button
                  className="btn ghost camera-tool-btn"
                  onClick={switchCamera}
                  title="Switch Front/Rear Camera"
                >
                  🔄 Flip Camera
                </button>
              )}
              <button
                className="camera-shutter-btn"
                onClick={triggerCapture}
                disabled={!!error || isStarting}
                title="Capture Photo"
              >
                <div className="shutter-inner" />
              </button>
              <button className="btn ghost camera-cancel-btn" onClick={onClose}>
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
