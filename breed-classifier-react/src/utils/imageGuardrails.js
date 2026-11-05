const GRADIENT_THRESHOLD = 48;
const ANGLE_TOLERANCE = 4;
const BLOCK_SIZE = 16;
const FLAT_BLOCK_SIGMA = 3.0;
const CENTER_BOX = { x0: 40, y0: 40, x1: 220, y1: 220 };
const CENTER_STRAIGHT_EDGE_MAX = 0.30;
const CENTER_FLAT_PATCH_MAX = 0.30;
const CENTER_FLAT_EDGE_MIN = 6000;
const GLOBAL_STRAIGHT_EDGE_MAX = 0.55;
const GLOBAL_FLAT_PATCH_MAX = 0.70;
const GLOBAL_FLAT_EDGE_MIN = 3000;
const ENTROPY_MAX = 0.86;
const TOP3_MASS_MIN = 0.20;
const TOP5_MASS_MIN = 0.30;
const BINARY_FLOOR_MIN = 0.60;
const MSP_FLOOR_MIN = 0.5863;
const ENERGY_FLOOR_MIN = -3.2004;
const FACE_AREA_MIN = 0.035;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function regionStats(gray, width, height, x0, y0, x1, y1) {
  const left = Math.max(1, Math.floor(x0));
  const top = Math.max(1, Math.floor(y0));
  const right = Math.min(width - 1, Math.ceil(x1));
  const bottom = Math.min(height - 1, Math.ceil(y1));
  let strongEdges = 0;
  let straightEdges = 0;

  for (let y = top; y < bottom; y += 1) {
    const row = y * width;
    for (let x = left; x < right; x += 1) {
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
  for (let by = Math.max(0, Math.floor(y0)); by + BLOCK_SIZE <= bottom; by += BLOCK_SIZE) {
    for (let bx = Math.max(0, Math.floor(x0)); bx + BLOCK_SIZE <= right; bx += BLOCK_SIZE) {
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

  const area = Math.max(1, (right - left) * (bottom - top));
  return {
    strongEdges,
    edgeDensity: strongEdges / area,
    straightEdgeRatio: strongEdges ? straightEdges / strongEdges : 0,
    flatPatchRatio: blockCount ? flatBlocks / blockCount : 0,
  };
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

  const full = regionStats(gray, width, height, 0, 0, width, height);
  const center = regionStats(
    gray,
    width,
    height,
    CENTER_BOX.x0,
    CENTER_BOX.y0,
    CENTER_BOX.x1,
    CENTER_BOX.y1,
  );

  const details = {
    strong_edges: full.strongEdges,
    straight_edge_ratio: Number(full.straightEdgeRatio.toFixed(4)),
    flat_patch_ratio: Number(full.flatPatchRatio.toFixed(4)),
    center_strong_edges: center.strongEdges,
    center_straight_edge_ratio: Number(center.straightEdgeRatio.toFixed(4)),
    center_flat_patch_ratio: Number(center.flatPatchRatio.toFixed(4)),
    center_edge_density: Number(center.edgeDensity.toFixed(4)),
    skin_ratio: Number((count ? skinPixels / count : 0).toFixed(4)),
    green_ratio: Number((count ? greenPixels / count : 0).toFixed(4)),
  };

  const centerIsManMade =
    center.straightEdgeRatio > CENTER_STRAIGHT_EDGE_MAX ||
    (center.flatPatchRatio > CENTER_FLAT_PATCH_MAX && center.strongEdges < CENTER_FLAT_EDGE_MIN);
  const globalIsManMade =
    full.straightEdgeRatio > GLOBAL_STRAIGHT_EDGE_MAX ||
    (full.flatPatchRatio > GLOBAL_FLAT_PATCH_MAX && full.strongEdges < GLOBAL_FLAT_EDGE_MIN);

  if (centerIsManMade || globalIsManMade) {
    return { rejected: true, reason_code: "manmade_object_or_screen", details };
  }
  return { rejected: false, details };
}

function buildSkinMap(data, width, height) {
  const count = width * height;
  const skin = new Uint8Array(count);
  const luma = new Float32Array(count);
  const skinLuma = [];

  for (let i = 0, p = 0; i < count; i += 1, p += 4) {
    const r = data[p];
    const g = data[p + 1];
    const b = data[p + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    luma[i] = y;

    const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
    if (y > 60 && cb >= 80 && cb <= 125 && cr >= 136 && cr <= 172 && r > g && g > b && r - g >= 12) {
      skin[i] = 1;
      skinLuma.push(y);
    }
  }
  return { skin, luma, skinLuma };
}

async function detectFace(canvas) {
  if (typeof window === "undefined" || !("FaceDetector" in window)) {
    return { available: false, count: 0, maxArea: 0 };
  }
  try {
    const detector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 3 });
    const faces = await detector.detect(canvas);
    const maxArea = faces.reduce((largest, face) => {
      const box = face.boundingBox || {};
      const area = (Number(box.width) * Number(box.height)) / (canvas.width * canvas.height);
      return Math.max(largest, Number.isFinite(area) ? area : 0);
    }, 0);
    return { available: true, count: faces.length, maxArea };
  } catch {
    return { available: false, count: 0, maxArea: 0 };
  }
}

export async function checkHumanSelfie(canvas) {
  if (!canvas || !canvas.width || !canvas.height) {
    return { detected: false, details: { skipped: "no_canvas" } };
  }

  const face = await detectFace(canvas);
  if (face.count > 0 && face.maxArea >= FACE_AREA_MIN) {
    return {
      detected: true,
      details: {
        tier: "face_detector",
        face_count: face.count,
        face_area_ratio: Number(face.maxArea.toFixed(4)),
        face_detector_available: true,
      },
    };
  }

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const { width, height } = canvas;
  const { data } = ctx.getImageData(0, 0, width, height);
  const { skin, luma, skinLuma } = buildSkinMap(data, width, height);

  let centerSkin = 0;
  let centerPixels = 0;
  for (let y = 55; y < 205; y += 1) {
    const row = y * width;
    for (let x = 55; x < 205; x += 1) {
      centerSkin += skin[row + x];
      centerPixels += 1;
    }
  }
  const centerSkinRatio = centerPixels ? centerSkin / centerPixels : 0;

  const sortedSkinLuma = [...skinLuma].sort((a, b) => a - b);
  const medianSkinLuma = sortedSkinLuma.length
    ? sortedSkinLuma[Math.floor(sortedSkinLuma.length / 2)]
    : 0;
  const darkCut = Math.min(85, medianSkinLuma * 0.62);
  const medianSkinLumaSafe = medianSkinLuma;

  const localSigmas = [];
  for (let y = 27; y < 213; y += 2) {
    const row = y * width;
    for (let x = 47; x < 213; x += 2) {
      if (!skin[row + x]) continue;
      let sum = 0;
      let sq = 0;
      for (let yy = -2; yy <= 2; yy += 1) {
        const nearby = (y + yy) * width + x;
        for (let xx = -2; xx <= 2; xx += 1) {
          const value = luma[nearby + xx];
          sum += value;
          sq += value * value;
        }
      }
      const mean = sum / 25;
      localSigmas.push(Math.sqrt(Math.max(0, sq / 25 - mean * mean)));
    }
  }
  localSigmas.sort((a, b) => a - b);
  const medianSigma = localSigmas.length
    ? localSigmas[Math.floor(localSigmas.length / 2)]
    : 0;

  let bestSymmetry = 0;
  for (let shift = -24; shift <= 24; shift += 4) {
    const axis = 130 + shift;
    let both = 0;
    let matches = 0;
    for (let y = 50; y < 200; y += 2) {
      const row = y * width;
      for (let x = 55; x < 205; x += 2) {
        const mirrored = 2 * axis - x;
        if (mirrored < 45 || mirrored > 214) continue;
        const a = skin[row + x];
        const b = skin[row + mirrored];
        if (!a && !b) continue;
        both += 1;
        if (a === b) matches += 1;
      }
    }
    if (both) {
      const score = matches / both;
      if (score > bestSymmetry) bestSymmetry = score;
    }
  }

  let darkCap = 0;
  let capPixels = 0;
  for (let y = 25; y < 100; y += 1) {
    const row = y * width;
    for (let x = 55; x < 205; x += 1) {
      capPixels += 1;
      if (luma[row + x] < darkCut) darkCap += 1;
    }
  }

  let leftDark = 0;
  let rightDark = 0;
  let eyePixels = 0;
  for (let y = 60; y < 140; y += 1) {
    const row = y * width;
    for (let x = 70; x < 190; x += 1) {
      eyePixels += 1;
      if (luma[row + x] < darkCut) {
        if (x < 130) leftDark += 1;
        else rightDark += 1;
      }
    }
  }

  const darkCapRatio = capPixels ? darkCap / capPixels : 0;
  const eyeBrowDensity = eyePixels ? (leftDark + rightDark) / eyePixels : 0;
  const eyeBrowBalance = Math.min(leftDark, rightDark) / Math.max(1, Math.max(leftDark, rightDark));
  const details = {
    tier: "ycrcb_fallback",
    face_count: face.count,
    face_area_ratio: Number(face.maxArea.toFixed(4)),
    face_detector_available: face.available,
    center_skin_ratio: Number(centerSkinRatio.toFixed(4)),
    median_skin_luma: Number(medianSkinLumaSafe.toFixed(2)),
    median_skin_sigma: Number(medianSigma.toFixed(2)),
    skin_symmetry: Number(bestSymmetry.toFixed(4)),
    dark_cap_ratio: Number(darkCapRatio.toFixed(4)),
    eye_brow_density: Number(eyeBrowDensity.toFixed(4)),
    eye_brow_balance: Number(eyeBrowBalance.toFixed(4)),
  };

  const portrait =
    centerSkinRatio > 0.62
    && medianSigma >= 2.2
    && medianSigma <= 14.5
    && bestSymmetry > 0.82
    && darkCapRatio > 0.02
    && eyeBrowDensity > 0.05;

  return { detected: portrait, details };
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

export async function evaluateImageGuardrails({
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
  const human = await checkHumanSelfie(canvas);
  if (human.detected) {
    return {
      rejected: true,
      reason_code: "human_or_selfie",
      stage: "human",
      details: human.details,
    };
  }

  const geometry = geometryGate(canvas);
  if (geometry.rejected) {
    return {
      rejected: true,
      reason_code: geometry.reason_code,
      stage: "geometry",
      details: { ...geometry.details, ...human.details },
    };
  }

  const entropy = normalizedEntropy(combinedProbs);
  const peak = logitPeak(activeLogits);
  const binaryConfidence = Math.max(...binaryProbs);
  const sortedProbs = [...combinedProbs].sort((a, b) => b - a);
  const top2Mass = sortedProbs.slice(0, 2).reduce((sum, value) => sum + value, 0);
  const top3Mass = sortedProbs.slice(0, 3).reduce((sum, value) => sum + value, 0);
  const top5Mass = sortedProbs.slice(0, 5).reduce((sum, value) => sum + value, 0);
  const binaryCutoff = Math.max(Number(binaryFloor) || 0, BINARY_FLOOR_MIN);
  const mspCutoff = Math.max(Number(mspThreshold) || 0, MSP_FLOOR_MIN);
  const energyCutoff = Math.max(Number(energyThreshold) || 0, ENERGY_FLOOR_MIN);

  const details = {
    ...geometry.details,
    ...human.details,
    normalized_entropy: Number(entropy.toFixed(4)),
    z_peak: Number(peak.zpeak.toFixed(4)),
    z_max: Number(peak.zmax.toFixed(4)),
    top1_prob: Number((sortedProbs[0] || 0).toFixed(4)),
    top2_mass: Number(top2Mass.toFixed(4)),
    top3_mass: Number(top3Mass.toFixed(4)),
    top5_mass: Number(top5Mass.toFixed(4)),
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
  if (top3Mass < TOP3_MASS_MIN || top5Mass < TOP5_MASS_MIN) {
    return { rejected: true, reason_code: "ood_low_topk_mass", stage: "logits", details };
  }

  return { rejected: false, reason_code: null, stage: "passed", details };
}
