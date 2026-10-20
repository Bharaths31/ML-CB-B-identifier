/**
 * Herd / session log persisted in localStorage.
 *
 * Each entry is intentionally small: a 160px thumbnail plus the prediction
 * fields needed by the CSV/JSON/ODT exports. The full result object is not
 * stored to keep the 5 MB localStorage budget safe.
 */

export const SESSION_KEY = "breed-herd-session-v1";
export const SESSION_LIMIT = 60;
export const THUMB_SIZE = 160;

function hasStorage() {
  try {
    return typeof window !== "undefined" && !!window.localStorage;
  } catch {
    return false;
  }
}

export function readSession() {
  if (!hasStorage()) return [];
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((e) => e && e.id) : [];
  } catch {
    return [];
  }
}

export function writeSession(entries) {
  if (!hasStorage()) return;
  try {
    window.localStorage.setItem(SESSION_KEY, JSON.stringify(entries.slice(0, SESSION_LIMIT)));
  } catch {
    // Quota exceeded: drop thumbnails and retry once.
    try {
      const lean = entries.slice(0, Math.floor(SESSION_LIMIT / 2)).map((e) => ({ ...e, thumbnail: null }));
      window.localStorage.setItem(SESSION_KEY, JSON.stringify(lean));
    } catch {
      /* give up silently; the in-memory session still works */
    }
  }
}

export function clearSession() {
  if (!hasStorage()) return;
  try {
    window.localStorage.removeItem(SESSION_KEY);
  } catch {
    /* ignore */
  }
}

export function thumbnailFromDataUrl(dataUrl, size = THUMB_SIZE) {
  if (!dataUrl || typeof document === "undefined") return Promise.resolve(null);
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      const scale = Math.max(size / img.width, size / img.height);
      const sw = size / scale;
      const sh = size / scale;
      ctx.drawImage(img, (img.width - sw) / 2, (img.height - sh) / 2, sw, sh, 0, 0, size, size);
      resolve(canvas.toDataURL("image/jpeg", 0.7));
    };
    img.onerror = () => resolve(null);
    img.src = dataUrl;
  });
}

export function makeSessionEntry({ result, quality, thumbnail, filename }) {
  const top5 = (result?.top5_breeds || result?.top_k || []).map((b) => ({
    breed: b.breed || "",
    confidence:
      typeof b.confidence === "number"
        ? b.confidence
        : typeof b.prob === "number"
        ? b.prob * 100
        : 0,
  }));
  const confidence = typeof result?.confidence === "number" ? result.confidence : 0;
  const species = result?.predicted_species || result?.species || "Unknown";
  return {
    id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    timestamp: new Date().toISOString(),
    filename: filename || result?.filename || "capture.jpg",
    species,
    speciesConfidence: result?.species_confidence ?? 0,
    breed: result?.predicted_breed || result?.top_breed || result?.breed || "Unknown",
    confidence,
    isOod: Boolean(result?.is_ood),
    oodScore: result?.ood_score ?? null,
    model: result?.model_name || result?.model_used || "unknown",
    top5,
    qualityScore: quality?.overallScore ?? null,
    qualityStatus: quality?.status ?? null,
    thumbnail: thumbnail || null,
  };
}

export function addSessionEntry(entry, entries = readSession()) {
  const next = [entry, ...entries].slice(0, SESSION_LIMIT);
  writeSession(next);
  return next;
}

export function sessionStats(entries) {
  const total = entries.length;
  const cattle = entries.filter((e) => String(e.species).toLowerCase() === "cattle").length;
  const buffalo = entries.filter((e) => String(e.species).toLowerCase() === "buffalo").length;
  const flagged = entries.filter((e) => e.isOod).length;
  // OOD entries deliberately carry zero confidence; averaging them in would
  // make a healthy session look uncertain. Report the recognised average and
  // surface flagged animals as a separate count.
  const recognised = entries.filter((e) => !e.isOod);
  const avgConfidence = recognised.length
    ? recognised.reduce((sum, e) => sum + (Number(e.confidence) || 0), 0) / recognised.length
    : 0;
  const avgQuality = entries.filter((e) => e.qualityScore != null).length
    ? entries.filter((e) => e.qualityScore != null).reduce((s, e) => s + e.qualityScore, 0) /
      entries.filter((e) => e.qualityScore != null).length
    : null;
  return { total, cattle, buffalo, flagged, avgConfidence, avgQuality };
}

function csvCell(value) {
  const text = value == null ? "" : String(value);
  if (/[",\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

export function sessionToCsv(entries) {
  const header = [
    "timestamp",
    "filename",
    "species",
    "species_confidence",
    "breed",
    "confidence",
    "is_ood",
    "ood_score",
    "quality_score",
    "model",
  ];
  const rows = entries.map((e) =>
    [
      e.timestamp,
      e.filename,
      e.species,
      e.speciesConfidence,
      e.breed,
      (e.confidence * 100).toFixed(2),
      e.isOod ? "yes" : "no",
      e.oodScore ?? "",
      e.qualityScore ?? "",
      e.model,
    ]
      .map(csvCell)
      .join(",")
  );
  return [header.join(","), ...rows].join("\n");
}

export function sessionToJson(entries) {
  return JSON.stringify(
    { exportedAt: new Date().toISOString(), count: entries.length, animals: entries },
    null,
    2
  );
}

/** Map session entries to the shape expected by /api/export-odt. */
export function sessionToOdtResults(entries) {
  return entries
    .slice()
    .reverse()
    .map((e) => ({
      filename: e.filename,
      species: e.species,
      predicted_species: e.species,
      species_confidence: e.speciesConfidence,
      top_breed: e.breed,
      predicted_breed: e.breed,
      top_breed_confidence: (Number(e.confidence) || 0) * 100,
      confidence: Number(e.confidence) || 0,
      top5_breeds: e.top5 || [],
      is_ood: e.isOod,
      ood: e.isOod,
      ood_score: e.oodScore,
      model_used: e.model,
      model_name: e.model,
    }));
}

export function downloadTextFile(text, filename, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
