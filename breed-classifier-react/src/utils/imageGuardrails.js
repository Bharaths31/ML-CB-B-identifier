const GRADIENT_THRESHOLD = 48;
const ANGLE_TOLERANCE = 4;
const STRAIGHT_EDGE_MAX = 0.33;
const BLOCK_SIZE = 16;
const FLAT_BLOCK_SIGMA = 3.0;
const FLAT_PATCH_MAX = 0.42;
const ENTROPY_MAX = 0.76;
const Z_PEAK_MIN = 2.0;
const Z_MAX_MIN = 1.6;
const BINARY_FLOOR_MIN = 0.60;
const MSP_FLOOR_MIN = 0.5863;
const ENERGY_FLOOR_MIN = -3.2004;
const HUMAN_SKIN_MAX = 0.70;
const HUMAN_SKIN_EDGE_MAX = 0.32;
const HUMAN_SKIN_GREEN_MAX = 0.02;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function geometryGate(canvas) {
  if (!canvas || !canvas.width || !canvas.height) {
    return { rejected: false, details: { skipped: "no_canvas" } };
  }

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const { width, height } = canvas;
  const { data } = ctx.getImageData(0, 0, width, height);
  const count = width * height;
  const gray = new Float32Array(count);
  let skinPixels = 0;
  let greenPixels = 0;

  for (let i = 0, p = 0; i < count; i += 1, p += 4) {
    const r = data[p] / 255;
    const g = data[p + 1] / 255;
    const b = data[p + 2] / 255;
    gray[i] = 0.2126 * data[p] + 0.7152 * data[p + 1] + 0.0722 * data[p + 2];

    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const cb = -0.168736 * r - 0.331264 * g + 0.5 * b;
    const cr = 0.5 * r - 0.418688 * g - 0.081312 * b;
    if (y > 0.25 && cb >= -0.15 && cb <= 0.1 && cr >= 0.02 && cr <= 0.22) {
      skinPixels += 1;
    }
    if (g > r * 1.05 && g > b * 1.05) greenPixels += 1;
  }

  let strongEdges = 0;
  let straightEdges = 0;

  for (let y = 1; y < height - 1; y += 1) {
    const row = y * width;
    for (let x = 1; x < width - 1; x += 1) {
      const i = row + x;
      const gx =
        -gray[i - width - 1] + gray[i - width + 1]
        - 2 * gray[i - 1] + 2 * gray[i + 1]
        - gray[i + width - 1] + gray[i + width + 1];
      const gy =
        -gray[i - width - 1] - 2 * gray[i - width] - gray[i - width + 1]
        + gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1];
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      if (magnitude <= GRADIENT_THRESHOLD) continue;

      strongEdges += 1;
      const theta = Math.abs((Math.atan2(gy, gx) * 180) / Math.PI);
      if (
        theta <= ANGLE_TOLERANCE ||
        Math.abs(theta - 90) <= ANGLE_TOLERANCE ||
        Math.abs(theta - 180) <= ANGLE_TOLERANCE
      ) {
        straightEdges += 1;
      }
    }
  }

  let flatBlocks = 0;
  let blockCount = 0;
  for (let by = 0; by + BLOCK_SIZE <= height; by += BLOCK_SIZE) {
    for (let bx = 0; bx + BLOCK_SIZE <= width; bx += BLOCK_SIZE) {
      let sum = 0;
      let sq = 0;
      for (let yy = 0; yy < BLOCK_SIZE; yy += 1) {
        const row = (by + yy) * width + bx;
        for (let xx = 0; xx < BLOCK_SIZE; xx += 1) {
          const value = gray[row + xx];
          sum += value;
          sq += value * value;
        }
      }
      const mean = sum / (BLOCK_SIZE * BLOCK_SIZE);
      const variance = Math.max(0, sq / (BLOCK_SIZE * BLOCK_SIZE) - mean * mean);
      if (Math.sqrt(variance) < FLAT_BLOCK_SIGMA) flatBlocks += 1;
      blockCount += 1;
    }
  }

  const straightEdgeRatio = strongEdges ? straightEdges / strongEdges : 0;
  const flatPatchRatio = blockCount ? flatBlocks / blockCount : 0;
  const edgeDensity = count ? strongEdges / count : 0;
  const skinRatio = count ? skinPixels / count : 0;
  const greenRatio = count ? greenPixels / count : 0;
  const details = {
    strong_edges: strongEdges,
    straight_edge_ratio: Number(straightEdgeRatio.toFixed(4)),
    flat_patch_ratio: Number(flatPatchRatio.toFixed(4)),
    skin_ratio: Number(skinRatio.toFixed(4)),
    green_ratio: Number(greenRatio.toFixed(4)),
  };

  if (
    straightEdgeRatio > STRAIGHT_EDGE_MAX ||
    flatPatchRatio > 0.55 ||
    (flatPatchRatio > FLAT_PATCH_MAX && strongEdges < 6000) ||
    (
      skinRatio > HUMAN_SKIN_MAX &&
      edgeDensity < HUMAN_SKIN_EDGE_MAX &&
      greenRatio < HUMAN_SKIN_GREEN_MAX
    )
  ) {
    return { rejected: true, reason_code: "manmade_object_or_screen", details };
  }
  return { rejected: false, details };
}

function normalizedEntropy(probabilities) {
  const total = probabilities.reduce((acc, value) => acc + value, 0) || 1;
  let entropy = 0;
  for (let i = 0; i < probabilities.length; i += 1) {
    const p = clamp(probabilities[i] / total, 1e-12, 1);
    entropy -= p * Math.log(p);
  }
  return entropy / Math.log(probabilities.length || 2);
}

function logitPeak(logits) {
  if (!logits || !logits.length) {
    return { zmax: -Infinity, mean: 0, std: 0, zpeak: 0 };
  }
  let max = -Infinity;
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const value = Number(logits[i]);
    if (value > max) max = value;
    sum += value;
  }
  const mean = sum / logits.length;
  let variance = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const delta = Number(logits[i]) - mean;
    variance += delta * delta;
  }
  const std = Math.sqrt(variance / logits.length);
  return {
    zmax: max,
    mean,
    std,
    zpeak: (max - mean) / (std + 1e-6),
  };
}

export function evaluateImageGuardrails({
  canvas,
  combinedProbs,
  activeLogits,
  binaryProbs,
  energy,
  msp,
  energyThreshold,
  mspThreshold,
  binaryFloor,
  oodEnabled = true,
}) {
  const geometry = geometryGate(canvas);
  if (geometry.rejected) {
    return {
      rejected: true,
      reason_code: geometry.reason_code,
      stage: "geometry",
      details: geometry.details,
    };
  }

  const entropy = normalizedEntropy(combinedProbs);
  const peak = logitPeak(activeLogits);
  const binaryConfidence = Math.max(...binaryProbs);
  const binaryCutoff = Math.max(Number(binaryFloor) || 0, BINARY_FLOOR_MIN);
  const mspCutoff = Math.max(Number(mspThreshold) || 0, MSP_FLOOR_MIN);
  const energyCutoff = Math.max(Number(energyThreshold) || 0, ENERGY_FLOOR_MIN);

  const details = {
    ...geometry.details,
    normalized_entropy: Number(entropy.toFixed(4)),
    z_peak: Number(peak.zpeak.toFixed(4)),
    z_max: Number(peak.zmax.toFixed(4)),
    binary_confidence: Number(binaryConfidence.toFixed(4)),
    binary_cutoff: binaryCutoff,
    energy: Number(Number(energy).toFixed(4)),
    energy_threshold: energyCutoff,
    msp: Number(Number(msp).toFixed(4)),
    msp_threshold: mspCutoff,
    ood_enabled: Boolean(oodEnabled),
  };

  if (!oodEnabled) {
    return { rejected: false, reason_code: null, stage: "passed", details };
  }

  if (binaryConfidence < binaryCutoff) {
    return { rejected: true, reason_code: "ood_low_binary_confidence", stage: "logits", details };
  }
  if (Number(energy) > energyCutoff) {
    return { rejected: true, reason_code: "ood_energy", stage: "logits", details };
  }
  if (Number(msp) < mspCutoff) {
    return { rejected: true, reason_code: "ood_msp", stage: "logits", details };
  }
  if (entropy > ENTROPY_MAX) {
    return { rejected: true, reason_code: "ood_entropy", stage: "logits", details };
  }
  if (peak.zpeak < Z_PEAK_MIN || peak.zmax < Z_MAX_MIN) {
    return { rejected: true, reason_code: "ood_peak", stage: "logits", details };
  }

  return { rejected: false, reason_code: null, stage: "passed", details };
}
