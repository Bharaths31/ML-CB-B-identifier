// Thin API layer — talks to the Flask backend (port 5000 via the Vite proxy).
// The standalone test_model.py GUI on 8501 is a separate application.

async function readJsonResponse(response, fallbackMessage) {
  let data = {};
  try {
    data = await response.json();
  } catch {
    data = {};
  }
  if (!response.ok || data.error) {
    const error = new Error(
      data.error || `${fallbackMessage} (HTTP ${response.status})`
    );
    error.status = response.status;
    error.payload = data;
    throw error;
  }
  return data;
}

export async function getModels() {
  const r = await fetch("/api/models");
  return readJsonResponse(r, "Failed to load models");
}

export async function predict(file, model, filename) {
  const form = new FormData();
  form.append("image", file);
  form.append("model", model);
  form.append("filename", filename);
  const r = await fetch("/api/predict", { method: "POST", body: form });
  const text = await r.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`Prediction failed (HTTP ${r.status})`);
  }
  if (!r.ok || data.error) {
    throw new Error(data.error || `Prediction failed (HTTP ${r.status})`);
  }
  return data;
}

export async function getDevInfo() {
  const r = await fetch("/api/dev-info");
  return readJsonResponse(r, "Failed to load breed class maps");
}

export async function getModelSpec(modelName) {
  const r = await fetch("/api/model-spec?model=" + encodeURIComponent(modelName));
  return readJsonResponse(r, "Failed to load model specification");
}

export async function getImageInfo(file, filename) {
  const form = new FormData();
  form.append("image", file);
  form.append("filename", filename);
  const r = await fetch("/api/image-info", { method: "POST", body: form });
  return readJsonResponse(r, "Failed to read image metadata");
}

export async function getPresenterConfig() {
  const r = await fetch("/api/presenter-config");
  return readJsonResponse(r, "Failed to load presenter config");
}

export async function savePresenterConfig(cfg) {
  const r = await fetch("/api/presenter-config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  return readJsonResponse(r, "Failed to save presenter config");
}

// --- Shared Configuration DB API ---
export async function getSharedConfig() {
  const r = await fetch("/api/config");
  if (!r.ok) throw new Error("Failed to load configuration");
  return r.json();
}

export async function updateSharedConfig(patchOrKeyVal) {
  const r = await fetch("/api/config", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patchOrKeyVal),
  });
  if (!r.ok) throw new Error("Failed to update configuration");
  return r.json();
}

export async function resetSharedConfig() {
  const r = await fetch("/api/config/reset", { method: "POST" });
  if (!r.ok) throw new Error("Failed to reset configuration");
  return r.json();
}

// --- OOD Management API ---
export async function getOODProfiles() {
  const r = await fetch("/api/ood/profiles");
  if (!r.ok) throw new Error("Failed to load OOD profiles");
  return r.json();
}

export async function getActiveOOD() {
  const r = await fetch("/api/ood/active");
  if (!r.ok) throw new Error("Failed to get active OOD");
  return r.json();
}

export async function selectOODProfile(profile) {
  const r = await fetch("/api/ood/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile }),
  });
  return readJsonResponse(r, "Failed to select OOD profile");
}

export async function toggleOOD(enabled) {
  const r = await fetch("/api/ood/toggle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  return readJsonResponse(r, "Failed to toggle OOD");
}

export async function updateOODThresholds(thresholds) {
  const r = await fetch("/api/ood/thresholds", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(thresholds),
  });
  return readJsonResponse(r, "Failed to update OOD thresholds");
}

export async function uploadOODProfile(file) {
  const form = new FormData();
  form.append("file", file);
  const r = await fetch("/api/ood/upload", { method: "POST", body: form });
  return readJsonResponse(r, "Failed to upload OOD profile");
}

export async function getOODDataDirs() {
  const r = await fetch("/api/ood/data-dirs");
  if (!r.ok) throw new Error("Failed to load calibration folders");
  return r.json();
}

export async function recalibrateOOD(payload) {
  const r = await fetch("/api/ood/recalibrate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.error) {
    throw new Error(data.error || "OOD recalibration failed");
  }
  return data;
}

// --- Model Selection & Upload ---
export async function selectModel(modelName) {
  const r = await fetch("/api/select-model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: modelName }),
  });
  return readJsonResponse(r, "Failed to switch model");
}

export async function uploadModel(file, filename) {
  const form = new FormData();
  form.append("model_file", file);
  form.append("filename", filename || file.name);
  const r = await fetch("/api/upload-model", { method: "POST", body: form });
  return readJsonResponse(r, "Model upload failed");
}

export async function getLogs() {
  const r = await fetch("/api/logs");
  if (!r.ok) throw new Error(`Failed to load logs (HTTP ${r.status})`);
  return r.text();
}

export async function exportOdt(results, mode) {
  const r = await fetch("/api/export-odt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ results, mode }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.error || "Export failed");
  }
  return r.blob();
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// --- Expert escalation / active-learning queue ---
export async function createEscalation(payload) {
  const r = await fetch("/api/escalations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJsonResponse(r, "Escalation queue failed");
}

export async function getEscalations(status = "all") {
  const r = await fetch(`/api/escalations?status=${encodeURIComponent(status)}`);
  return readJsonResponse(r, "Failed to load escalations");
}

export async function resolveEscalation(id, payload) {
  const r = await fetch(`/api/escalations/${encodeURIComponent(id)}/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJsonResponse(r, "Could not resolve escalation");
}

// --- Bharat Pashudhan (BPA) QR matrix ---
export async function getBpaQrMatrix(data) {
  const r = await fetch("/api/bpa/qr", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data }),
  });
  const body = await r.json().catch(() => ({}));
  if (!r.ok || body.error) throw new Error(body.error || "QR generation failed");
  return body;
}
