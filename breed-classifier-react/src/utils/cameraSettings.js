/**
 * Persistent camera options shared between CameraCapture and AdvancedPanel.
 */

export const CAMERA_SETTINGS_KEY = "breed_classifier_camera_settings";

export const DEFAULT_CAMERA_SETTINGS = {
  facingMode: "environment", // "environment" | "user"
  autoCapture: false,
  burstMode: false,
  silhouette: "off", // "off" | "cattle" | "buffalo"
  silhouetteFlipped: false,
  torchDefault: false,
  clahe: false,
};

export function getCameraSettings() {
  try {
    const raw = localStorage.getItem(CAMERA_SETTINGS_KEY);
    if (raw) {
      return { ...DEFAULT_CAMERA_SETTINGS, ...JSON.parse(raw) };
    }
  } catch {
    /* localStorage may be restricted in sandbox */
  }
  return { ...DEFAULT_CAMERA_SETTINGS };
}

export function saveCameraSettings(settings) {
  try {
    const current = getCameraSettings();
    const updated = { ...current, ...settings };
    localStorage.setItem(CAMERA_SETTINGS_KEY, JSON.stringify(updated));
    return updated;
  } catch {
    return settings;
  }
}

