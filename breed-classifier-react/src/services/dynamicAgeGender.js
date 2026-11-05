import {
  AGE_OPTIONS,
  EXPECTED_BUFFALO_BREEDS,
  EXPECTED_CATTLE_BREEDS,
} from "./breedCatalog.js";

const COMPACT_BREEDS = new Set([
  "punganur", "vechur", "malnad_gidda", "bargur", "badri", "motu",
  "khariar", "siri", "toda",
]);

const HEAVY_BREEDS = new Set([
  "kankrej", "jaffarabadi", "ongole", "kangayam", "amritmahal", "hallikar",
  "khillari", "krishna_valley", "murrah", "nili_ravi",
]);

const DRAUGHT_BREEDS = new Set([
  "amritmahal", "hallikar", "kangayam", "khillari", "ongole", "nagori",
  "kankrej", "jaffarabadi",
]);

const MILCH_BREEDS = new Set([
  "gir", "sahiwal", "red_sindhi", "rathi", "tharparkar", "murrah", "surti",
  "mehsana",
]);

const SAMPLE = 130;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0) || 1;
  return exps.map((value) => value / sum);
}

function wholeFrameFeatures(canvas) {
  const sample = document.createElement("canvas");
  sample.width = SAMPLE;
  sample.height = SAMPLE;
  const ctx = sample.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(canvas, 0, 0, SAMPLE, SAMPLE);
  const { data } = ctx.getImageData(0, 0, SAMPLE, SAMPLE);
  const gray = new Float32Array(SAMPLE * SAMPLE);

  for (let i = 0, p = 0; i < gray.length; i += 1, p += 4) {
    gray[i] = 0.2126 * data[p] + 0.7152 * data[p + 1] + 0.0722 * data[p + 2];
  }

  const centerStart = Math.floor(SAMPLE * 0.15);
  const centerEnd = Math.ceil(SAMPLE * 0.85);
  let fineEnergy = 0;
  let fineCount = 0;
  let coarseEnergy = 0;
  let coarseCount = 0;

  for (let y = centerStart + 1; y < centerEnd - 1; y += 1) {
    for (let x = centerStart + 1; x < centerEnd - 1; x += 1) {
      const i = y * SAMPLE + x;
      const laplacian =
        -4 * gray[i] + gray[i - 1] + gray[i + 1] + gray[i - SAMPLE] + gray[i + SAMPLE];
      fineEnergy += Math.abs(laplacian);
      fineCount += 1;

      let coarse = 0;
      for (let yy = -4; yy <= 4; yy += 1) {
        for (let xx = -4; xx <= 4; xx += 1) {
          coarse += gray[i + yy * SAMPLE + xx];
        }
      }
      coarse /= 81;
      const gx = gray[i + 1] - gray[i - 1];
      const gy = gray[i + SAMPLE] - gray[i - SAMPLE];
      coarseEnergy += Math.abs(gx) + Math.abs(gy) + Math.abs(coarse - gray[i]) * 0.5;
      coarseCount += 1;
    }
  }

  const fineMean = fineEnergy / Math.max(1, fineCount);
  const coarseMean = coarseEnergy / Math.max(1, coarseCount);
  const S_texture = clamp((fineMean / (coarseMean * 0.35 + 1)) * 0.5, 0, 1);

  const bandStats = (y0, y1) => {
    let sum = 0;
    let sq = 0;
    let edge = 0;
    let count = 0;
    for (let y = y0; y < y1; y += 1) {
      for (let x = 1; x < SAMPLE - 1; x += 1) {
        const i = y * SAMPLE + x;
        const value = gray[i];
        sum += value;
        sq += value * value;
        edge += Math.abs(gray[i + 1] - gray[i - 1]) + Math.abs(gray[i + SAMPLE] - gray[i - SAMPLE]);
        count += 1;
      }
    }
    const mean = sum / Math.max(1, count);
    const variance = Math.max(0, sq / Math.max(1, count) - mean * mean);
    return { variance, edge: edge / Math.max(1, count) };
  };

  const upper = bandStats(Math.floor(SAMPLE * 0.08), Math.floor(SAMPLE * 0.38));
  const lower = bandStats(Math.floor(SAMPLE * 0.38), Math.floor(SAMPLE * 0.82));
  const S_bulk = clamp(
    (lower.variance + lower.edge * 4) /
      (upper.variance + upper.edge * 4 + lower.variance + lower.edge * 4 + 1),
    0,
    1
  );

  const quadrantEnergy = (x0, x1, y0, y1) => {
    const startX = Math.floor(SAMPLE * x0);
    const endX = Math.ceil(SAMPLE * x1);
    const startY = Math.floor(SAMPLE * y0);
    const endY = Math.ceil(SAMPLE * y1);
    let energy = 0;
    let count = 0;
    for (let y = Math.max(1, startY); y < Math.min(SAMPLE - 1, endY); y += 1) {
      for (let x = Math.max(1, startX); x < Math.min(SAMPLE - 1, endX); x += 1) {
        const i = y * SAMPLE + x;
        energy += Math.abs(gray[i + 1] - gray[i - 1]) + Math.abs(gray[i + SAMPLE] - gray[i - SAMPLE]);
        count += 1;
      }
    }
    return energy / Math.max(1, count);
  };

  const anterior = Math.max(
    quadrantEnergy(0.05, 0.5, 0.08, 0.5),
    quadrantEnergy(0.5, 0.95, 0.08, 0.5)
  );
  const posterior = Math.max(
    quadrantEnergy(0.05, 0.5, 0.5, 0.92),
    quadrantEnergy(0.5, 0.95, 0.5, 0.92)
  );
  const anteriorPosterior = clamp(
    (anterior - posterior) / (anterior + posterior + 1e-6),
    -1,
    1
  );

  return {
    S_texture,
    S_bulk,
    anterior_posterior: anteriorPosterior,
    fine_energy: fineMean,
    coarse_energy: coarseMean,
  };
}

function standardizedMeans({ logitsBySlug, topIndex, allClasses, compactSet, heavySet }) {
  let mean = 0;
  let sq = 0;
  const values = allClasses.map((slug, index) => {
    const value = Number(logitsBySlug.get(String(slug).toLowerCase()) ?? 0);
    mean += value;
    sq += value * value;
    return { index, slug: String(slug).toLowerCase(), value };
  });
  mean /= Math.max(1, values.length);
  const variance = Math.max(0, sq / Math.max(1, values.length) - mean * mean);
  const std = Math.sqrt(variance) || 1;

  const average = (set) => {
    let total = 0;
    let count = 0;
    for (const item of values) {
      if (item.index === topIndex) continue;
      if (!set.has(item.slug)) continue;
      total += (item.value - mean) / std;
      count += 1;
    }
    return count ? total / count : 0;
  };

  return {
    compact: average(compactSet),
    heavy: average(heavySet),
    draught: average(DRAUGHT_BREEDS),
    milch: average(MILCH_BREEDS),
  };
}

export function estimateDynamicAgeGender({
  canvas,
  breedSlug,
  cattleLogits = [],
  buffaloLogits = [],
  cattleClasses = EXPECTED_CATTLE_BREEDS,
  buffaloClasses = EXPECTED_BUFFALO_BREEDS,
}) {
  const allClasses = [...cattleClasses, ...buffaloClasses];
  const allLogits = [...cattleLogits, ...buffaloLogits];
  const logitsBySlug = new Map();
  for (let i = 0; i < allClasses.length; i += 1) {
    logitsBySlug.set(String(allClasses[i] || "").toLowerCase(), Number(allLogits[i]) || 0);
  }
  const topIndex = allClasses.findIndex(
    (slug) => String(slug).toLowerCase() === String(breedSlug || "").toLowerCase()
  );
  const projections = standardizedMeans({
    logitsBySlug,
    topIndex,
    allClasses,
    compactSet: COMPACT_BREEDS,
    heavySet: HEAVY_BREEDS,
  });

  const features = wholeFrameFeatures(canvas);
  const S_neoteny = sigmoid(1.8 * (projections.compact - projections.heavy));
  const S_texture = features.S_texture;
  const S_bulk = features.S_bulk;
  const S_dimorph = sigmoid(
    1.5 * (projections.draught - projections.milch) +
    1.2 * features.anterior_posterior
  );

  const ageLogits = [
    2.4 * S_neoteny - 1.6 * S_bulk - 1.4 * S_texture,
    1.8 * (1 - Math.abs(S_bulk - 0.5) * 2) +
      0.8 * (1 - Math.abs(S_neoteny - 0.5) * 2) -
      0.4 * S_texture,
    2.2 * S_bulk + 1.8 * S_texture - 1.8 * S_neoteny,
  ];
  const ageProbs = softmax(ageLogits);
  const ageKeys = ["calf", "young", "adult"];
  let ageIndex = 0;
  for (let i = 1; i < ageProbs.length; i += 1) {
    if (ageProbs[i] > ageProbs[ageIndex]) ageIndex = i;
  }
  const ageKey = ageKeys[ageIndex];
  const ageLabel = AGE_OPTIONS.find((item) => item.key === ageKey)?.label || ageKey;

  const genderLogits = [
    2.5 * (1 - S_dimorph) + 0.6 * S_bulk - 0.8,
    2.5 * S_dimorph + 0.8 * S_texture - 1.1,
  ];
  const genderProbs = softmax(genderLogits);
  const female = genderProbs[0] >= genderProbs[1];
  const genderConfidence = Math.round(
    Math.max(genderProbs[0], genderProbs[1]) * 100
  );
  const genderLabel = ageKey === "calf"
    ? `${female ? "Female" : "Male"} Calf`
    : female ? "Female" : "Male";

  return {
    age_group: {
      key: ageKey,
      label: ageLabel,
      confidence: Math.round(ageProbs[ageIndex] * 100),
      method: "dynamic logit + whole-frame moments",
      signals: {
        S_neoteny: Number(S_neoteny.toFixed(4)),
        S_texture: Number(S_texture.toFixed(4)),
        S_bulk: Number(S_bulk.toFixed(4)),
        age_logits: ageLogits.map((value) => Number(value.toFixed(4))),
      },
    },
    gender: {
      key: female ? "female" : "male",
      label: genderLabel,
      confidence: genderConfidence,
      method: "dynamic dimorphism logits + anterior/posterior energy",
      signals: {
        S_dimorph: Number(S_dimorph.toFixed(4)),
        anterior_posterior: Number(features.anterior_posterior.toFixed(4)),
        gender_logits: genderLogits.map((value) => Number(value.toFixed(4))),
      },
    },
    features: {
      ...features,
      compact_projection: Number(projections.compact.toFixed(4)),
      heavy_projection: Number(projections.heavy.toFixed(4)),
      draught_projection: Number(projections.draught.toFixed(4)),
      milch_projection: Number(projections.milch.toFixed(4)),
    },
  };
}
