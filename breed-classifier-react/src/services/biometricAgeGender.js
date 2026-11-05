import {
  AGE_OPTIONS,
  DAIRY_BREEDS,
  EXPECTED_BUFFALO_BREEDS,
  EXPECTED_CATTLE_BREEDS,
} from "./breedCatalog.js";

const SMALL_COMPACT_BREEDS = new Set([
  "punganur",
  "vechur",
  "malnad_gidda",
  "kasargod",
  "bargur",
  "badri",
  "motu",
  "khariar",
  "bhadwari",
]);

const HEAVY_MATURE_BREEDS = new Set([
  "kankrej",
  "jaffarabadi",
  "ongole",
  "kangayam",
  "amritmahal",
  "hallikar",
  "khillari",
  "krishna_valley",
  "murrah",
  "nili_ravi",
]);

const DRAUGHT_BREEDS = new Set([
  "amritmahal",
  "hallikar",
  "kangayam",
  "khillari",
  "ongole",
  "nagori",
]);

const SAMPLE_SIZE = 80;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function softmax(values, temperature = 1) {
  const scaled = values.map((value) => value / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - max));
  const sum = exps.reduce((acc, value) => acc + value, 0) || 1;
  return exps.map((value) => value / sum);
}

function sumExpBySlug(slugs, logitsBySlug, temperature = 2) {
  let sum = 0;
  for (const slug of slugs) {
    const logit = logitsBySlug.get(slug);
    if (Number.isFinite(logit)) sum += Math.exp(logit / temperature);
  }
  return sum;
}

function analyzeBody(canvas) {
  if (!canvas || !canvas.width || !canvas.height) {
    return {
      torsoFill: 0.56,
      crownEnergy: 0.45,
      compactRatio: 1,
      neckRatio: 1,
      foregroundRatio: 0,
      bbox: null,
    };
  }

  const sample = document.createElement("canvas");
  sample.width = SAMPLE_SIZE;
  sample.height = SAMPLE_SIZE;
  const ctx = sample.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(canvas, 0, 0, SAMPLE_SIZE, SAMPLE_SIZE);
  const { data } = ctx.getImageData(0, 0, SAMPLE_SIZE, SAMPLE_SIZE);

  const borderLimit = Math.max(2, Math.round(SAMPLE_SIZE * 0.1));
  let borderR = 0;
  let borderG = 0;
  let borderB = 0;
  let borderCount = 0;
  let borderSq = 0;

  for (let y = 0; y < SAMPLE_SIZE; y += 1) {
    for (let x = 0; x < SAMPLE_SIZE; x += 1) {
      const isBorder =
        x < borderLimit || x >= SAMPLE_SIZE - borderLimit ||
        y < borderLimit || y >= SAMPLE_SIZE - borderLimit;
      if (!isBorder) continue;
      const p = (y * SAMPLE_SIZE + x) * 4;
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];
      borderR += r;
      borderG += g;
      borderB += b;
      borderSq += (r + g + b) / 3;
      borderCount += 1;
    }
  }

  borderR /= borderCount || 1;
  borderG /= borderCount || 1;
  borderB /= borderCount || 1;
  const borderMean = borderSq / (borderCount || 1);
  const borderStd = Math.sqrt(
    Math.max(
      0,
      data.reduce((acc, _, index) => {
        if (index % 4 !== 0) return acc;
        const x = (index / 4) % SAMPLE_SIZE;
        const y = Math.floor(index / 4 / SAMPLE_SIZE);
        const isBorder =
          x < borderLimit || x >= SAMPLE_SIZE - borderLimit ||
          y < borderLimit || y >= SAMPLE_SIZE - borderLimit;
        if (!isBorder) return acc;
        const value = (data[index] + data[index + 1] + data[index + 2]) / 3;
        return acc + (value - borderMean) * (value - borderMean);
      }, 0) / (borderCount || 1)
    )
  );

  const mask = new Uint8Array(SAMPLE_SIZE * SAMPLE_SIZE);
  const gray = new Float32Array(SAMPLE_SIZE * SAMPLE_SIZE);
  const saturation = new Float32Array(SAMPLE_SIZE * SAMPLE_SIZE);

  let minX = SAMPLE_SIZE;
  let maxX = -1;
  let minY = SAMPLE_SIZE;
  let maxY = -1;
  let foreground = 0;

  for (let y = 0; y < SAMPLE_SIZE; y += 1) {
    for (let x = 0; x < SAMPLE_SIZE; x += 1) {
      const index = y * SAMPLE_SIZE + x;
      const p = index * 4;
      const r = data[p];
      const g = data[p + 1];
      const b = data[p + 2];
      const yValue = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      gray[index] = yValue;
      const maxChannel = Math.max(r, g, b);
      const minChannel = Math.min(r, g, b);
      const sat = maxChannel ? (maxChannel - minChannel) / maxChannel : 0;
      saturation[index] = sat;

      const distance = Math.hypot(r - borderR, g - borderG, b - borderB);
      const centerWeight = 1 - 0.22 * Math.abs(x / SAMPLE_SIZE - 0.5)
        - 0.22 * Math.abs(y / SAMPLE_SIZE - 0.5);
      const isForeground =
        (distance > Math.max(24, borderStd * 1.15) ||
          (sat > 0.34 && distance > 16)) &&
        centerWeight > 0.7;

      if (isForeground) {
        mask[index] = 1;
        foreground += 1;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < minX || maxY < minY || foreground < SAMPLE_SIZE * SAMPLE_SIZE * 0.04) {
    return {
      torsoFill: 0.56,
      crownEnergy: 0.45,
      compactRatio: 1,
      neckRatio: 1,
      foregroundRatio: foreground / (SAMPLE_SIZE * SAMPLE_SIZE),
      bbox: null,
    };
  }

  const bboxWidth = maxX - minX + 1;
  const bboxHeight = maxY - minY + 1;
  const area = bboxWidth * bboxHeight;

  const regionFill = (x0, x1, y0, y1) => {
    let total = 0;
    let filled = 0;
    const startX = Math.floor(minX + bboxWidth * x0);
    const endX = Math.ceil(minX + bboxWidth * x1);
    const startY = Math.floor(minY + bboxHeight * y0);
    const endY = Math.ceil(minY + bboxHeight * y1);
    for (let y = startY; y < endY; y += 1) {
      for (let x = startX; x < endX; x += 1) {
        total += 1;
        if (mask[y * SAMPLE_SIZE + x]) filled += 1;
      }
    }
    return total ? filled / total : 0;
  };

  const torsoFill = regionFill(0.14, 0.86, 0.42, 0.82);
  const topFill = regionFill(0.1, 0.9, 0, 0.3);
  const bottomFill = regionFill(0.1, 0.9, 0.72, 1);
  const topLeft = regionFill(0, 0.5, 0, 0.5);
  const topRight = regionFill(0.5, 1, 0, 0.5);
  const bottomLeft = regionFill(0, 0.5, 0.5, 1);
  const bottomRight = regionFill(0.5, 1, 0.5, 1);

  let topEdge = 0;
  let topEdgeCount = 0;
  let allEdge = 0;
  let allEdgeCount = 0;
  const topLimit = Math.floor(minY + bboxHeight * 0.3);
  for (let y = Math.max(1, minY); y < Math.min(SAMPLE_SIZE - 1, maxY); y += 1) {
    for (let x = Math.max(1, minX); x < Math.min(SAMPLE_SIZE - 1, maxX); x += 1) {
      const i = y * SAMPLE_SIZE + x;
      const gx = -gray[i - SAMPLE_SIZE - 1] + gray[i - SAMPLE_SIZE + 1]
        - 2 * gray[i - 1] + 2 * gray[i + 1]
        - gray[i + SAMPLE_SIZE - 1] + gray[i + SAMPLE_SIZE + 1];
      const gy = -gray[i - SAMPLE_SIZE - 1] - 2 * gray[i - SAMPLE_SIZE]
        - gray[i - SAMPLE_SIZE + 1] + gray[i + SAMPLE_SIZE - 1]
        + 2 * gray[i + SAMPLE_SIZE] + gray[i + SAMPLE_SIZE + 1];
      const magnitude = Math.abs(gx) + Math.abs(gy);
      allEdge += magnitude;
      allEdgeCount += 1;
      if (y <= topLimit) {
        topEdge += magnitude;
        topEdgeCount += 1;
      }
    }
  }

  const crownEnergy = clamp(
    (topEdge / Math.max(1, topEdgeCount)) /
      (allEdge / Math.max(1, allEdgeCount) * 1.25 + 1),
    0,
    1
  );
  const neckRatio = (Math.max(topLeft, topRight) + 0.02) /
    (Math.max(bottomLeft, bottomRight) + 0.02);

  return {
    torsoFill: clamp(torsoFill, 0, 1),
    crownEnergy,
    compactRatio: bboxWidth / Math.max(1, bboxHeight),
    neckRatio,
    topFill,
    bottomFill,
    foregroundRatio: foreground / area,
    bbox: { x: minX, y: minY, width: bboxWidth, height: bboxHeight },
  };
}

function juvenileLogitIndex({ topSlug, logitsBySlug }) {
  const smallMass = sumExpBySlug(SMALL_COMPACT_BREEDS, logitsBySlug);
  const heavyMass = sumExpBySlug(HEAVY_MATURE_BREEDS, logitsBySlug);
  let index = smallMass / (smallMass + heavyMass + 1e-9);
  if (SMALL_COMPACT_BREEDS.has(topSlug)) index *= 0.55;
  else if (HEAVY_MATURE_BREEDS.has(topSlug)) index *= 0.85;
  return clamp(index, 0, 1);
}

export function estimateBiometricAgeGender({
  canvas,
  breedSlug,
  species,
  cattleLogits = [],
  buffaloLogits = [],
  cattleClasses = EXPECTED_CATTLE_BREEDS,
  buffaloClasses = EXPECTED_BUFFALO_BREEDS,
  combinedProbs = [],
}) {
  const body = analyzeBody(canvas);
  const allClasses = [...cattleClasses, ...buffaloClasses];
  const allLogits = [...cattleLogits, ...buffaloLogits];
  const logitsBySlug = new Map();
  const probsBySlug = new Map();
  for (let i = 0; i < allClasses.length; i += 1) {
    const slug = String(allClasses[i] || "").toLowerCase();
    if (!slug) continue;
    logitsBySlug.set(slug, Number(allLogits[i]) || 0);
    probsBySlug.set(slug, Number(combinedProbs[i]) || 0);
  }

  const juvenile = juvenileLogitIndex({
    topSlug: String(breedSlug || "").toLowerCase(),
    logitsBySlug,
  });
  const torso = body.torsoFill;
  const crown = body.crownEnergy;

  const rawScores = [
    0.40 * juvenile + 0.35 * (1 - torso) + 0.25 * (1 - crown),
    1.0 - Math.abs(torso - 0.56) * 2.2 + 0.20 * (1 - Math.abs(juvenile - 0.5)),
    0.40 * torso + 0.35 * crown + 0.25 * (1 - juvenile),
  ];
  const ageProbs = softmax(rawScores, 1.05);
  const ageKeys = ["calf", "young", "adult"];
  let ageIndex = 0;
  for (let i = 1; i < ageProbs.length; i += 1) {
    if (ageProbs[i] > ageProbs[ageIndex]) ageIndex = i;
  }
  const ageKey = ageKeys[ageIndex];
  const ageConfidence = Math.round(clamp(65 + ageProbs[ageIndex] * 29, 65, 94));
  const ageLabel = AGE_OPTIONS.find((item) => item.key === ageKey)?.label || ageKey;

  const massFor = (set) => {
    let sum = 0;
    for (const slug of set) sum += probsBySlug.get(slug) || 0;
    return sum;
  };
  const bullMass = massFor(DRAUGHT_BREEDS);
  const dairyMass = massFor(DAIRY_BREEDS);
  const breedBias = (bullMass - dairyMass) / (bullMass + dairyMass + 1e-9);
  const morphology = clamp((body.neckRatio - 1) * 1.8, -0.35, 0.35);
  const femaleScore = clamp(
    0.5 - 0.38 * breedBias - 0.18 * morphology + (torso > 0.62 ? 0.03 : 0),
    0.05,
    0.95
  );
  const female = femaleScore >= 0.5;
  const genderConfidence = Math.round(
    clamp(65 + Math.abs(femaleScore - 0.5) * 2 * 29, 65, 92)
  );

  return {
    gender: {
      key: female ? "female" : "male",
      label: female ? "Female" : "Male",
      confidence: genderConfidence,
      method: "breed logit prior + body mass ratio",
      signals: {
        bull_mass: Number(bullMass.toFixed(4)),
        dairy_mass: Number(dairyMass.toFixed(4)),
        neck_ratio: Number(body.neckRatio.toFixed(3)),
      },
    },
    age_group: {
      key: ageKey,
      label: ageLabel,
      confidence: ageConfidence,
      method: "biometric logit + body proportion fusion",
      signals: {
        juvenile_logit_index: Number(juvenile.toFixed(4)),
        torso_fill: Number(torso.toFixed(4)),
        crown_energy: Number(crown.toFixed(4)),
        compact_ratio: Number(body.compactRatio.toFixed(3)),
        foreground_ratio: Number(body.foregroundRatio.toFixed(3)),
      },
    },
    body,
    species,
  };
}
