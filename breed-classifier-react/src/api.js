// Thin API layer — talks to the test_model.py backend (default port 8501).
// Vite proxies /api -> http://localhost:8501 during development.

export async function getModels() {
  const r = await fetch("/api/models");
  if (!r.ok) throw new Error("Failed to load models");
  return r.json();
}

export async function predict(file, model, filename) {
  const form = new FormData();
  form.append("image", file);
  form.append("model", model);
  form.append("filename", filename);
  const r = await fetch("/api/predict", { method: "POST", body: form });
  return r.json();
}

export async function getDevInfo() {
  const r = await fetch("/api/dev-info");
  return r.json();
}

export async function getModelSpec(modelName) {
  const r = await fetch("/api/model-spec?model=" + encodeURIComponent(modelName));
  return r.json();
}

export async function getImageInfo(file, filename) {
  const form = new FormData();
  form.append("image", file);
  form.append("filename", filename);
  const r = await fetch("/api/image-info", { method: "POST", body: form });
  return r.json();
}

export async function getPresenterConfig() {
  const r = await fetch("/api/presenter-config");
  return r.json();
}

export async function savePresenterConfig(cfg) {
  const r = await fetch("/api/presenter-config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  return r.json();
}

export async function getLogs() {
  const r = await fetch("/api/logs");
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
