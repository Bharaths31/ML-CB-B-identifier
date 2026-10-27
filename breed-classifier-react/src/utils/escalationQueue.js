/**
 * Durable, offline-first outbox for veterinary escalations.
 *
 * Cases are queued in localStorage when the field worker has no signal and are
 * replayed (automatically on socket reconnect, or manually with one click)
 * against POST /api/escalations.
 */

import { createEscalation } from "../api.js";

export const OUTBOX_KEY = "breed-escalation-outbox-v1";
export const OUTBOX_EVENT = "breed-escalation-outbox-changed";

function storageAvailable() {
  try {
    return typeof window !== "undefined" && !!window.localStorage;
  } catch {
    return false;
  }
}

function notify() {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(OUTBOX_EVENT));
  }
}

export function readOutbox() {
  if (!storageAvailable()) return [];
  try {
    const parsed = JSON.parse(window.localStorage.getItem(OUTBOX_KEY) || "[]");
    return Array.isArray(parsed) ? parsed.filter((entry) => entry && entry.payload) : [];
  } catch {
    return [];
  }
}

export function writeOutbox(entries) {
  if (!storageAvailable()) return [];
  try {
    window.localStorage.setItem(OUTBOX_KEY, JSON.stringify(entries.slice(0, 100)));
  } catch {
    /* quota: keep the newest half without images */
    try {
      const lean = entries.slice(0, 50).map((entry) => ({
        ...entry,
        payload: { ...entry.payload, image_b64: null, attention_map_b64: null },
      }));
      window.localStorage.setItem(OUTBOX_KEY, JSON.stringify(lean));
    } catch {
      /* give up silently */
    }
  }
  notify();
  return readOutbox();
}

export function enqueueEscalation(payload) {
  const entry = {
    localId: `local_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
    queuedAt: new Date().toISOString(),
    payload,
  };
  const next = [entry, ...readOutbox()];
  writeOutbox(next);
  return entry;
}

function dequeue(localId) {
  writeOutbox(readOutbox().filter((entry) => entry.localId !== localId));
}

export function outboxCount() {
  return readOutbox().length;
}

/**
 * Try to flush every queued case. Successful entries are removed; transient
 * failures remain for the next reconnect. Permanent 4xx validation failures
 * are dropped so one malformed case cannot block the outbox forever.
 */
export async function syncOutbox(onProgress) {
  const entries = readOutbox();
  if (!entries.length) return { synced: 0, remaining: 0, dropped: 0, errors: [] };

  let synced = 0;
  let dropped = 0;
  const errors = [];
  for (const entry of entries) {
    try {
      await createEscalation(entry.payload);
      dequeue(entry.localId);
      synced += 1;
      onProgress?.({ synced, remaining: entries.length - synced });
    } catch (err) {
      const status = Number(err?.status) || 0;
      if (status >= 400 && status < 500 && status !== 408 && status !== 429) {
        dequeue(entry.localId);
        dropped += 1;
        errors.push({ localId: entry.localId, error: err?.message || "rejected", dropped: true });
      } else {
        errors.push({ localId: entry.localId, error: err?.message || "offline" });
      }
    }
  }
  return { synced, dropped, remaining: readOutbox().length, errors };
}

/** True when the error is a permanent client-side rejection, not offline. */
export function isPermanentEscalationError(error) {
  const status = Number(error?.status) || 0;
  return status >= 400 && status < 500 && status !== 408 && status !== 429;
}

/** Ask the browser for a GPS fix; resolves null instead of blocking a case. */
export function getBrowserGps(timeoutMs = 4500) {
  if (typeof navigator === "undefined" || !navigator.geolocation) {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    let settled = false;
    const done = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    const timer = setTimeout(() => done(null), timeoutMs);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        clearTimeout(timer);
        done({
          gps_lat: position.coords.latitude,
          gps_lng: position.coords.longitude,
          location_label: `GPS ±${Math.round(position.coords.accuracy || 0)} m`,
        });
      },
      () => {
        clearTimeout(timer);
        done(null);
      },
      { enableHighAccuracy: true, timeout: timeoutMs, maximumAge: 60000 }
    );
  });
}

export function buildEscalationPayload({ result, imageB64, quality, filename, gps }) {
  const top3 = (result?.top_k || result?.top5_breeds || []).slice(0, 3).map((item) => ({
    breed: item.breed || item.name || "",
    prob:
      typeof item.prob === "number"
        ? item.prob
        : typeof item.confidence === "number"
        ? item.confidence / 100
        : 0,
  }));
  return {
    filename: filename || result?.filename || "capture.jpg",
    image_b64: imageB64 || null,
    gps_lat: gps?.gps_lat ?? null,
    gps_lng: gps?.gps_lng ?? null,
    location_label: gps?.location_label || "Manual capture",
    flw_id: "FLW-TN-SALEM-01",
    model_used: result?.model_name || result?.model_used || "unknown",
    predicted_species: result?.predicted_species || result?.species || "Unknown",
    predicted_breed: result?.predicted_breed || result?.top_breed || "Unknown",
    confidence: Number(result?.confidence) || 0,
    confidence_level: Number(result?.confidence_level) || 1,
    is_ood: Boolean(result?.is_ood || result?.ood),
    ood_details: result?.ood_details || null,
    top3,
    quality: quality || null,
    attention_map_b64: result?.attention_map_b64 || null,
  };
}
