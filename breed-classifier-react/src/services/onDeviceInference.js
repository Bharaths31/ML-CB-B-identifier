/**
 * On-device inference + persistent IndexedDB model manager.
 *
 * This module is intentionally framework-free: the React pages call
 * `runOnDeviceInference()` and `SettingsModal` calls the storage helpers.
 *
 * ONNX Runtime Web runs the exact same graph as the Python export:
 *   input  : float32 NCHW [1, 3, 260, 260]
 *   output : binary [1,2], cattle [1,50], buffalo [1,18]
 *
 * If the ONNX file is a `_MobileOutputs` export (name contains mobile/int8 or
 * metadata.normalizationBaked=true), pixels are fed as [0,1]. Otherwise the
 * caller-side ImageNet normalization from src/config.py is applied.
 */

import * as ort from "onnxruntime-web/wasm";
import {
  DEFAULT_OOD,
  EXPECTED_BUFFALO_BREEDS,
  EXPECTED_CATTLE_BREEDS,
  IMAGE_SIZE,
  IMAGENET_MEAN,
  IMAGENET_STD,
  breedDisplayName,
} from "./breedCatalog.js";
import { ENV_BACKEND_URL, resolveApiBase } from "./apiBase.js";
import { evaluateImageGuardrails } from "../utils/imageGuardrails.js";
import { estimateDynamicAgeGender } from "./dynamicAgeGender.js";

const DB_NAME = "BreedClassifierDB";
const DB_VERSION = 1;
const MODELS_STORE = "models_store";
const OOD_STORE = "ood_store";
const SETTINGS_STORE = "settings_store";
const DEFAULT_MODEL_NAME = "default_model.onnx";
const BASE_URL = import.meta.env.BASE_URL || "/";

if (!import.meta.env.DEV) {
  // Production/Capacitor uses the extern-WASM build + public/ort files.
  // In dev ORT uses the asset Vite emits for its bundled build.
  ort.env.wasm.wasmPaths = `${BASE_URL}ort/`;
}
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.logLevel = "error";

export const DEFAULT_SETTINGS = {
  activeModel: DEFAULT_MODEL_NAME,
  oodEnabled: true,
  activeOodProfile: "auto",
  onlineMode: false,
  theme: "emerald-obsidian",
  language: "en",
  backendUrl: ENV_BACKEND_URL,
  developerMode: false,
  showConfidence: true,
  topBreedsCount: 3,
  showPredictionView: true,
  energyThreshold: DEFAULT_OOD.energyThreshold,
  mspThreshold: DEFAULT_OOD.mspThreshold,
  binaryConfidenceFloor: DEFAULT_OOD.binaryConfidenceFloor,
  energyTemperature: DEFAULT_OOD.energyTemperature,
};

// ---------------------------------------------------------------------------
// IndexedDB
// ---------------------------------------------------------------------------

let dbPromise = null;

function openDB() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB is not available in this WebView."));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(MODELS_STORE)) {
        db.createObjectStore(MODELS_STORE, { keyPath: "name" });
      }
      if (!db.objectStoreNames.contains(OOD_STORE)) {
        db.createObjectStore(OOD_STORE, { keyPath: "filename" });
      }
      if (!db.objectStoreNames.contains(SETTINGS_STORE)) {
        db.createObjectStore(SETTINGS_STORE, { keyPath: "key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("IndexedDB open failed"));
  });
  return dbPromise;
}

function txStore(db, store, mode = "readonly") {
  return db.transaction(store, mode).objectStore(store);
}

function requestToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function getAll(store) {
  const db = await openDB();
  return requestToPromise(txStore(db, store).getAll());
}

async function put(store, value) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, "readwrite");
    tx.objectStore(store).put(value);
    tx.oncomplete = () => resolve(value);
    tx.onerror = () => reject(tx.error);
  });
}

async function remove(store, key) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, "readwrite");
    tx.objectStore(store).delete(key);
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => reject(tx.error);
  });
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export async function getSettings() {
  try {
    const rows = await getAll(SETTINGS_STORE);
    const saved = Object.fromEntries(rows.map((row) => [row.key, row.value]));
    return { ...DEFAULT_SETTINGS, ...saved };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export async function saveSettings(patch) {
  const entries = Object.entries(patch || {});
  await Promise.all(entries.map(([key, value]) => put(SETTINGS_STORE, { key, value })));
  return getSettings();
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

function inferNormalizationBaked(name = "", metadata = {}) {
  if (typeof metadata.normalizationBaked === "boolean") return metadata.normalizationBaked;
  return /mobile|int8|baked/i.test(String(name));
}

// ---------------------------------------------------------------------------
// Minimal ONNX metadata reader (ModelProto.metadata_props = field 14)
// ---------------------------------------------------------------------------

function readVarint(buffer, offset) {
  let result = 0;
  let shift = 0;
  let byte = 0;
  do {
    byte = buffer[offset++];
    result |= (byte & 0x7f) << shift;
    shift += 7;
  } while (byte & 0x80 && offset < buffer.length);
  return [result >>> 0, offset];
}

function readProtoFields(buffer) {
  const fields = [];
  let offset = 0;
  while (offset < buffer.length) {
    let tag;
    [tag, offset] = readVarint(buffer, offset);
    const field = tag >>> 3;
    const wire = tag & 7;
    if (wire === 0) {
      let value;
      [value, offset] = readVarint(buffer, offset);
      fields.push({ field, wire, value });
    } else if (wire === 1) {
      offset += 8;
    } else if (wire === 2) {
      let length;
      [length, offset] = readVarint(buffer, offset);
      fields.push({ field, wire, value: buffer.subarray(offset, offset + length) });
      offset += length;
    } else if (wire === 5) {
      offset += 4;
    } else {
      break;
    }
  }
  return fields;
}

export function parseOnnxMetadata(arrayBuffer) {
  try {
    const buffer = new Uint8Array(arrayBuffer);
    const decoder = new TextDecoder();
    const metadata = {};
    for (const entry of readProtoFields(buffer)) {
      if (entry.field !== 14 || entry.wire !== 2) continue;
      let key = "";
      let value = "";
      for (const part of readProtoFields(entry.value)) {
        if (part.field === 1 && part.wire === 2) key = decoder.decode(part.value);
        if (part.field === 2 && part.wire === 2) value = decoder.decode(part.value);
      }
      if (key) metadata[key] = value;
    }
    for (const key of [
      "ood", "cattle_classes", "buffalo_classes", "imagenet_mean",
      "imagenet_std", "normalization_baked", "image_size", "eval_resize",
      "num_cattle", "num_buffalo", "model_meta_version", "backbone",
      "attention", "pad_to_square", "input_dtype", "class_map_source",
    ]) {
      if (typeof metadata[key] === "string") {
        try { metadata[key] = JSON.parse(metadata[key]); } catch { /* keep string */ }
      }
    }
    return metadata;
  } catch {
    return {};
  }
}

export async function listModels() {
  const models = await getAll(MODELS_STORE).catch(() => []);
  return models
    .map((model) => ({
      ...model,
      size_mb: Number(model.size_mb || (model.arrayBuffer?.byteLength || 0) / (1024 * 1024)),
      normalizationBaked: inferNormalizationBaked(model.name, model.metadata || {}),
      runnable: model.metadata?.runnable !== false,
    }))
    .sort((a, b) => {
      if (a.name === DEFAULT_MODEL_NAME) return -1;
      if (b.name === DEFAULT_MODEL_NAME) return 1;
      return String(b.uploaded_at || "").localeCompare(String(a.uploaded_at || ""));
    });
}

export async function getModel(name) {
  const db = await openDB();
  return requestToPromise(txStore(db, MODELS_STORE).get(name));
}

export async function ensureDefaultModel() {
  const existing = await getModel(DEFAULT_MODEL_NAME).catch(() => null);
  if (existing && existing.metadata?.ood) return existing;
  try {
    const response = await fetch(`${BASE_URL}models/${DEFAULT_MODEL_NAME}`);
    if (!response.ok) return existing || null;
    const arrayBuffer = await response.arrayBuffer();
    const embedded = parseOnnxMetadata(arrayBuffer);
    const embeddedOod = embedded.ood && typeof embedded.ood === "object" ? embedded.ood : null;
    const record = {
      name: DEFAULT_MODEL_NAME,
      arrayBuffer,
      size_mb: arrayBuffer.byteLength / (1024 * 1024),
      uploaded_at: new Date().toISOString(),
      metadata: {
        source: "bundled",
        normalizationBaked: typeof embedded.normalization_baked === "boolean"
          ? embedded.normalization_baked
          : false,
        imageSize: Number(embedded.image_size) || IMAGE_SIZE,
        numCattle: EXPECTED_CATTLE_BREEDS.length,
        numBuffalo: EXPECTED_BUFFALO_BREEDS.length,
        cattleClasses: EXPECTED_CATTLE_BREEDS,
        buffaloClasses: EXPECTED_BUFFALO_BREEDS,
        ood: embeddedOod,
        oodSource: embeddedOod ? "embedded" : "none",
        embedded,
      },
    };
    await put(MODELS_STORE, record);
    return record;
  } catch {
    return existing || null;
  }
}

export async function saveModelFile(file) {
  let name = file?.name || `uploaded_model_${Date.now()}.onnx`;
  let ext = (name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  if (!ext) {
    name = `${name}.onnx`;
    ext = ".onnx";
  }
  if (![".onnx", ".pt", ".pth"].includes(ext)) {
    throw new Error("Choose an .onnx model (or a .pt checkpoint to keep for later export).");
  }
  const arrayBuffer = await file.arrayBuffer();
  const runnable = ext === ".onnx";
  const embedded = runnable ? parseOnnxMetadata(arrayBuffer) : {};
  const embeddedOod = embedded.ood && typeof embedded.ood === "object" ? embedded.ood : null;
  const record = {
    name,
    arrayBuffer,
    size_mb: arrayBuffer.byteLength / (1024 * 1024),
    uploaded_at: new Date().toISOString(),
    metadata: {
      source: "uploaded",
      runnable,
      format: runnable ? "onnx" : "pytorch",
      reason: runnable ? null : "PyTorch .pt checkpoints must be exported to ONNX before on-device inference.",
      normalizationBaked: typeof embedded.normalization_baked === "boolean"
        ? embedded.normalization_baked
        : inferNormalizationBaked(name),
      imageSize: Number(embedded.image_size) || IMAGE_SIZE,
      padToSquare: Boolean(embedded.pad_to_square),
      numCattle: Number(embedded.num_cattle) || EXPECTED_CATTLE_BREEDS.length,
      numBuffalo: Number(embedded.num_buffalo) || EXPECTED_BUFFALO_BREEDS.length,
      cattleClasses: Array.isArray(embedded.cattle_classes) ? embedded.cattle_classes : EXPECTED_CATTLE_BREEDS,
      buffaloClasses: Array.isArray(embedded.buffalo_classes) ? embedded.buffalo_classes : EXPECTED_BUFFALO_BREEDS,
      ood: embeddedOod,
      oodSource: embeddedOod ? "embedded" : "none",
      embedded,
    },
  };
  if (runnable) {
    // Validate the graph before persisting it.
    await createSession(record);
  }
  await put(MODELS_STORE, record);
  return record;
}

export async function deleteModel(name) {
  if (name === DEFAULT_MODEL_NAME) return false;
  await remove(MODELS_STORE, name);
  return true;
}

// ---------------------------------------------------------------------------
// OOD profiles
// ---------------------------------------------------------------------------

function normalizeOodJson(raw) {
  const data = raw && typeof raw === "object" ? raw : {};
  const temperature = data.energy_temperature ?? data.temperature ?? DEFAULT_OOD.energyTemperature;
  return {
    energy_temperature: Number(temperature),
    energy_threshold: Number(data.energy_threshold ?? DEFAULT_OOD.energyThreshold),
    msp_threshold: Number(data.msp_threshold ?? DEFAULT_OOD.mspThreshold),
    binary_confidence_floor: Number(
      data.binary_confidence_floor ?? data.binary_floor ?? DEFAULT_OOD.binaryConfidenceFloor
    ),
    percentile: data.percentile ?? null,
    calibrated_at: data.calibrated_at ?? null,
    calibrated_on_samples: data.calibrated_on_samples ?? null,
  };
}

export async function listOodProfiles() {
  const profiles = await getAll(OOD_STORE).catch(() => []);
  return profiles.map((profile) => ({
    ...profile,
    json_data: normalizeOodJson(profile.json_data),
  }));
}

export async function getOodProfile(filename) {
  if (!filename || filename === "auto" || filename === "disabled") return null;
  const db = await openDB();
  const row = await requestToPromise(txStore(db, OOD_STORE).get(filename));
  return row ? { ...row, json_data: normalizeOodJson(row.json_data) } : null;
}

export async function saveOodProfile(filename, rawJson) {
  const clean = filename || `ood_${Date.now()}.json`;
  if (!/\.json$/i.test(clean)) throw new Error("OOD profile must be a .json file.");
  const json_data = normalizeOodJson(rawJson);
  const record = { filename: clean, json_data, uploaded_at: new Date().toISOString() };
  await put(OOD_STORE, record);
  return record;
}

export async function deleteOodProfile(filename) {
  await remove(OOD_STORE, filename);
  return true;
}

// ---------------------------------------------------------------------------
// ONNX Runtime Web
// ---------------------------------------------------------------------------

const sessionCache = new Map();

async function createSession(record) {
  const cached = sessionCache.get(record.name);
  if (cached) return cached;
  const bytes = record.arrayBuffer instanceof ArrayBuffer
    ? new Uint8Array(record.arrayBuffer)
    : record.arrayBuffer;
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  sessionCache.set(record.name, session);
  return session;
}

export async function getSession(record) {
  return createSession(record);
}

export function disposeSession(name) {
  const session = sessionCache.get(name);
  if (session?.release) {
    try { session.release(); } catch { /* ignore */ }
  }
  sessionCache.delete(name);
}

function softmax(values) {
  const arr = Array.from(values, Number);
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((v) => v / sum);
}

function logSumExp(values) {
  const arr = Array.from(values, Number);
  const max = Math.max(...arr);
  const sum = arr.reduce((acc, v) => acc + Math.exp(v - max), 0);
  return max + Math.log(sum || 1);
}

function tensorData(output) {
  if (!output) return null;
  if (output.data) return Array.from(output.data, Number);
  if (Array.isArray(output)) return output.map(Number);
  return null;
}

function resolveOutputs(session, results) {
  const names = session.outputNames || [];
  const byName = {};
  names.forEach((name) => { byName[String(name).toLowerCase()] = results[name]; });
  let binary = byName.binary || byName.output0;
  let cattle = byName.cattle;
  let buffalo = byName.buffalo;
  if (!binary || !cattle || !buffalo) {
    const ordered = names.map((name) => results[name]);
    if (!binary) binary = ordered[0];
    if (!cattle) cattle = ordered[1];
    if (!buffalo) buffalo = ordered[2];
  }
  return { binary, cattle, buffalo };
}

export function preprocessToTensor(canvasOrImage, normalizationBaked = false) {
  const size = IMAGE_SIZE;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(canvasOrImage, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  const tensor = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0, p = 0; i < plane; i += 1, p += 4) {
    const r = data[p] / 255;
    const g = data[p + 1] / 255;
    const b = data[p + 2] / 255;
    if (normalizationBaked) {
      tensor[i] = r;
      tensor[plane + i] = g;
      tensor[2 * plane + i] = b;
    } else {
      tensor[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      tensor[plane + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      tensor[2 * plane + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }
  return tensor;
}

function classListFromModel(record, key, fallback) {
  const meta = record?.metadata || {};
  const map = meta.classMaps || {};
  const value = meta[key] || map[key === "cattleClasses" ? "cattle" : "buffalo"];
  return Array.isArray(value) && value.length ? value : fallback;
}

export async function runOnDeviceInference({
  canvas,
  model,
  settings,
  oodProfile = null,
  imageMeta = {},
}) {
  const record = model || (await getModel(settings?.activeModel || DEFAULT_MODEL_NAME));
  if (!record) {
    throw new Error("No on-device model selected. Open Settings and upload an ONNX model.");
  }
  if (record.metadata?.runnable === false) {
    throw new Error(record.metadata?.reason || "This checkpoint is not runnable on-device. Export it to ONNX first.");
  }
  const session = await getSession(record);
  const normalizationBaked = inferNormalizationBaked(record.name, record.metadata || {});
  const inputName = session.inputNames?.[0] || "input";
  const tensorDataArray = preprocessToTensor(canvas, normalizationBaked);
  const input = new ort.Tensor("float32", tensorDataArray, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);

  const started = performance.now();
  const outputs = await session.run({ [inputName]: input });
  const { binary, cattle, buffalo } = resolveOutputs(session, outputs);
  const binaryLogits = tensorData(binary);
  const cattleLogits = tensorData(cattle);
  const buffaloLogits = tensorData(buffalo);
  if (!binaryLogits || !cattleLogits || !buffaloLogits) {
    throw new Error("Model outputs are missing binary/cattle/buffalo heads.");
  }
  const latencyMs = performance.now() - started;

  const binaryProbs = softmax(binaryLogits);
  const cattleProbs = softmax(cattleLogits);
  const buffaloProbs = softmax(buffaloLogits);
  const cattleClasses = classListFromModel(record, "cattleClasses", EXPECTED_CATTLE_BREEDS);
  const buffaloClasses = classListFromModel(record, "buffaloClasses", EXPECTED_BUFFALO_BREEDS);

  const combined = [
    ...cattleProbs.map((p) => binaryProbs[0] * p),
    ...buffaloProbs.map((p) => binaryProbs[1] * p),
  ];
  const ranked = combined
    .map((prob, index) => ({ prob, index }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5)
    .map(({ prob, index }) => {
      const isCattle = index < cattleClasses.length;
      const slug = isCattle ? cattleClasses[index] : buffaloClasses[index - cattleClasses.length];
      return {
        breed: breedDisplayName(slug || "unknown"),
        breed_slug: slug || "unknown",
        species: isCattle ? "Cattle" : "Buffalo",
        confidence: Math.round(prob * 10000) / 100,
        prob,
      };
    });

  const top = ranked[0] || { breed: "Unknown", breed_slug: null, species: "Cattle", confidence: 0, prob: 0 };
  const speciesIndex = binaryProbs[1] > binaryProbs[0] ? 1 : 0;
  const species = speciesIndex === 1 ? "Buffalo" : "Cattle";
  const speciesConfidence = Math.round(binaryProbs[speciesIndex] * 10000) / 100;

  // OOD: embedded metadata (mutated OOD model) > selected JSON > settings.
  const activeLogits = speciesIndex === 1 ? buffaloLogits : cattleLogits;
  const embeddedOod = record.metadata?.ood && typeof record.metadata.ood === "object"
    ? record.metadata.ood
    : null;
  const activeOod = embeddedOod || oodProfile?.json_data || null;
  const temperature = Number(
    activeOod?.energy_temperature ?? settings?.energyTemperature ?? DEFAULT_OOD.energyTemperature
  ) || 1;
  const energyThreshold = Number(
    activeOod?.energy_threshold ?? settings?.energyThreshold ?? DEFAULT_OOD.energyThreshold
  );
  const mspThreshold = Number(
    activeOod?.msp_threshold ?? settings?.mspThreshold ?? DEFAULT_OOD.mspThreshold
  );
  const binaryFloor = Number(
    activeOod?.binary_confidence_floor
      ?? settings?.binaryConfidenceFloor
      ?? DEFAULT_OOD.binaryConfidenceFloor
  );
  const energy = -temperature * logSumExp(
    [...binaryLogits, ...cattleLogits, ...buffaloLogits].map((value) => value / temperature)
  );
  const msp = Math.max(...binaryProbs);
  const binaryConfidence = Math.max(...binaryProbs);
  const hasCalibration = Boolean(embeddedOod || oodProfile)
    || record.metadata?.source === "bundled";
  const requiresCalibration = !hasCalibration && record.metadata?.source === "uploaded";
  const embeddedDisabled = embeddedOod?.enabled === false;
  const oodEnabled = settings?.oodEnabled !== false
    && settings?.activeOodProfile !== "disabled"
    && !embeddedDisabled
    && !requiresCalibration;
  const oodByFloor = binaryConfidence < binaryFloor;
  const oodByEnergy = energy > energyThreshold;
  const oodByMsp = msp < mspThreshold;
  const isOod = Boolean(oodEnabled && (oodByFloor || oodByEnergy || oodByMsp));

  const oodDetails = {
    energy: Number(energy.toFixed(4)),
    energy_threshold: energyThreshold,
    msp: Number(msp.toFixed(4)),
    msp_threshold: mspThreshold,
    binary_confidence: Number(binaryConfidence.toFixed(4)),
    binary_confidence_floor: binaryFloor,
    ood_by_energy: oodByEnergy,
    ood_by_msp: oodByMsp,
    ood_by_binary_floor: oodByFloor,
    temperature,
    profile: embeddedOod ? "embedded" : (oodProfile?.filename || settings?.activeOodProfile || "auto"),
    ood_source: embeddedOod ? "embedded" : (oodProfile?.filename ? "profile" : "defaults"),
    requires_calibration: requiresCalibration,
  };

  const guardrail = evaluateImageGuardrails({
    canvas,
    combinedProbs: combined,
    activeLogits,
    binaryProbs,
    energy,
    msp,
    energyThreshold,
    mspThreshold,
    binaryFloor,
    oodEnabled,
  });

  const baseResult = {
    model_name: record.name,
    model_source: record.metadata?.source || "uploaded",
    normalization_baked: normalizationBaked,
    inference_time_ms: Math.round(latencyMs * 10) / 10,
    total_time_ms: Math.round((performance.now() - started) * 10) / 10,
    routing: "on-device soft routing",
  };

  if (guardrail.rejected) {
    return {
      ...baseResult,
      rejected: true,
      reason_code: guardrail.reason_code,
      guardrail: guardrail.details,
      species: null,
      predicted_species: null,
      species_confidence: 0,
      species_prob: 0,
      breed: null,
      top_breed: null,
      breed_slug: null,
      confidence: 0,
      top_breed_confidence: 0,
      top5_breeds: [],
      gender: null,
      age_group: null,
      is_ood: true,
      ood: true,
      ood_score: Number(energy.toFixed(4)),
      ood_details: oodDetails,
      image_stats: {},
    };
  }

  const attrs = estimateDynamicAgeGender({
    canvas,
    breedSlug: top.breed_slug,
    cattleLogits,
    buffaloLogits,
    cattleClasses,
    buffaloClasses,
  });

  return {
    ...baseResult,
    rejected: false,
    guardrail: guardrail.details,
    species,
    predicted_species: species,
    species_confidence: speciesConfidence,
    species_prob: Math.max(...binaryProbs),
    breed: top.breed,
    top_breed: top.breed,
    breed_slug: top.breed_slug,
    confidence: top.confidence,
    top_breed_confidence: top.confidence,
    top5_breeds: ranked,
    gender: attrs.gender,
    age_group: attrs.age_group,
    is_ood: isOod,
    ood: isOod,
    ood_score: Number(energy.toFixed(4)),
    ood_details: oodDetails,
    image_stats: attrs.features,
  };
}

export async function fetchBreedKnowledgeOnline({
  settings,
  species,
  breedSlug,
  gender,
  ageGroup,
}) {
  if (!settings?.onlineMode || !navigator.onLine) {
    return { available: false, reason: "offline_mode" };
  }
  const base = resolveApiBase(settings.backendUrl);
  if (!base) {
    return { available: false, reason: "no_backend" };
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 7000);
  try {
    const knowledgeUrl = `${base}/api/breed-knowledge/${encodeURIComponent(species)}/${encodeURIComponent(breedSlug)}`;
    const knowledgeResponse = await fetch(knowledgeUrl, {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
    if (!knowledgeResponse.ok) {
      throw new Error(`Breed knowledge failed (HTTP ${knowledgeResponse.status})`);
    }
    const knowledgePayload = await knowledgeResponse.json();

    let advisory = null;
    try {
      const advisoryResponse = await fetch(`${base}/api/advisory-recalculate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          species,
          breed: breedSlug,
          gender: gender || null,
          age_group: ageGroup || null,
        }),
        signal: controller.signal,
      });
      if (advisoryResponse.ok) {
        const payload = await advisoryResponse.json();
        advisory = payload.advisory || null;
      }
    } catch {
      advisory = null;
    }

    return {
      available: true,
      knowledge: knowledgePayload.knowledge || knowledgePayload,
      advisory,
    };
  } finally {
    clearTimeout(timer);
  }
}

export async function verifyDynamicPredictions({
  canvases = [],
  settings,
  model = null,
} = {}) {
  const valid = canvases.filter((canvas) => canvas && canvas.width === IMAGE_SIZE && canvas.height === IMAGE_SIZE);
  const canvas260 = valid.length === canvases.length;

  if (valid.length < 2) {
    return {
      ok: false,
      canvas260,
      unique: false,
      silhouetteFree: typeof document === "undefined" || !document.querySelector(".camera-silhouette"),
      error: "Provide at least two 260x260 canvases.",
      results: [],
    };
  }

  const results = [];
  for (const canvas of valid.slice(0, 4)) {
    const result = await runOnDeviceInference({
      canvas,
      model,
      settings,
      imageMeta: { source: "verifyDynamicPredictions" },
    });
    results.push({
      rejected: result.rejected,
      reason_code: result.reason_code || null,
      age: result.age_group
        ? { key: result.age_group.key, confidence: result.age_group.confidence }
        : null,
      gender: result.gender
        ? { key: result.gender.key, confidence: result.gender.confidence }
        : null,
      breed: result.breed,
      confidence: result.confidence,
      ood_score: result.ood_score,
    });
  }

  const signatures = results.map((item) => JSON.stringify(item));
  const unique = new Set(signatures).size === signatures.length;
  const silhouetteFree =
    typeof document === "undefined" || !document.querySelector(".camera-silhouette");
  const report = { ok: canvas260 && unique && silhouetteFree, canvas260, unique, silhouetteFree, results };

  if (import.meta.env.DEV) {
    console.log("[verifyDynamicPredictions]", report);
  }
  return report;
}

export const ON_DEVICE_MODEL_NAME = DEFAULT_MODEL_NAME;
