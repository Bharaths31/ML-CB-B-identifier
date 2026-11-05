const GRADIENT_THRESHOLD = 45;
const AXIS_TOLERANCE = 4;
const AXIS_EDGE_RATIO_MAX = 0.32;
const EDGE_SPAN_MAX = 0.62;
const BLOCK_SIZE = 16;
const FLAT_BLOCK_STD = 3.2;
const FLAT_BLOCK_RATIO_MAX = 0.42;
const BLUE_DOMINANCE_MAX = 0.28;
const ENTROPY_MAX = 0.76;
const Z_PEAK_MIN = 2.0;
const Z_MAX_MIN = 1.6;
const BINARY_FLOOR_MIN = 0.72;
const MSP_FLOOR_MIN = 0.5863;
const ENERGY_FLOOR_MIN = -3.2004;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function luma(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function isAxisAligned(degrees) {
  const nearHorizontalGradient =
    degrees < AXIS_TOLERANCE ||
    degrees > 360 - AXIS_TOLERANCE ||
    Math.abs(degrees - 180) < AXIS_TOLERANCE;
  const nearVerticalGradient =
    Math.abs(degrees - 90) < AXIS_TOLERANCE ||
    Math.abs(degrees - 270) < AXIS_TOLERANCE;
  return {
    axis: nearHorizontalGradient || nearVerticalGradient,
    verticalLine: nearHorizontalGradient,
    horizontalLine: nearVerticalGradient,
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

  let sum = 0;
  let sumSq = 0;
  let bright = 0;
  let blueBright = 0;

  for (let i = 0, p = 0; i < count; i += 1, p += 4) {
    const r = data[p];
    const g = data[p + 1];
    const b = data[p + 2];
    const y = luma(r, g, b);
    gray[i] = y;
    sum += y;
    sumSq += y * y;
    if (y > 180) {
      bright += 1;
      if (b > r + 25) blueBright += 1;
    }
  }

  const meanLuma = sum / count;
  const variance = Math.max(0, sumSq / count - meanLuma * meanLuma);
  const blueDominance = bright ? blueBright / bright : 0;

  let strongEdges = 0;
  let axisEdges = 0;
  const currentColumnRun = new Uint16Array(width);
  const longestColumnRun = new Uint16Array(width);
  let longestRowRun = 0;
  let rowRun = 0;

  for (let y = 1; y < height - 1; y += 1) {
    const row = y * width;
    rowRun = 0;
    for (let x = 1; x < width - 1; x += 1) {
      const i = row + x;
      const gx =
        -gray[i - width - 1] + gray[i - width + 1]
        - 2 * gray[i - 1] + 2 * gray[i + 1]
        - gray[i + width - 1] + gray[i + width + 1];
      const gy =
        -gray[i - width - 1] - 2 * gray[i - width] - gray[i - width + 1]
        + gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1];
      const magnitude = Math.abs(gx) + Math.abs(gy);
      if (magnitude <= GRADIENT_THRESHOLD) {
        rowRun = 0;
        currentColumnRun[x] = 0;
        continue;
      }

      strongEdges += 1;
      let degrees = (Math.atan2(gy, gx) * 180) / Math.PI;
      if (degrees < 0) degrees += 360;
      const alignment = isAxisAligned(degrees);
      if (alignment.axis) {
        axisEdges += 1;
        if (alignment.horizontalLine) {
          rowRun += 1;
          if (rowRun > longestRowRun) longestRowRun = rowRun;
        } else {
          rowRun = 0;
        }
        if (alignment.verticalLine) {
          currentColumnRun[x] += 1;
          if (currentColumnRun[x] > longestColumnRun[x]) {
            longestColumnRun[x] = currentColumnRun[x];
          }
        } else {
          currentColumnRun[x] = 0;
        }
      } else {
        rowRun = 0;
        currentColumnRun[x] = 0;
      }
    }
  }

  let longestColumn = 0;
  for (let i = 0; i < longestColumnRun.length; i += 1) {
    if (longestColumnRun[i] > longestColumn) longestColumn = longestColumnRun[i];
  }
  const longestRow = longestRowRun;

  let flatBlocks = 0;
  let blockCount = 0;
  for (let by = 0; by + BLOCK_SIZE <= height; by += BLOCK_SIZE) {
    for (let bx = 0; bx + BLOCK_SIZE <= width; bx += BLOCK_SIZE) {
      let blockSum = 0;
      let blockSq = 0;
      for (let yy = 0; yy < BLOCK_SIZE; yy += 1) {
        const row = (by + yy) * width + bx;
        for (let xx = 0; xx < BLOCK_SIZE; xx += 1) {
          const value = gray[row + xx];
          blockSum += value;
          blockSq += value * value;
        }
      }
      const blockMean = blockSum / (BLOCK_SIZE * BLOCK_SIZE);
      const blockVar = Math.max(0, blockSq / (BLOCK_SIZE * BLOCK_SIZE) - blockMean * blockMean);
      if (Math.sqrt(blockVar) < FLAT_BLOCK_STD) flatBlocks += 1;
      blockCount += 1;
    }
  }

  const flatRatio = blockCount ? flatBlocks / blockCount : 0;
  const axisRatio = strongEdges ? axisEdges / strongEdges : 0;
  const rowSpan = longestRow / width;
  const columnSpan = longestColumn / height;

  const inner = 0.25;
  const outer = 0.08;
  let centerSum = 0;
  let centerCount = 0;
  let borderSum = 0;
  let borderCount = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = gray[y * width + x];
      const inCenter =
        x >= width * inner && x <= width * (1 - inner) &&
        y >= height * inner && y <= height * (1 - inner);
      const inBorder =
        x < width * outer || x > width * (1 - outer) ||
        y < height * outer || y > height * (1 - outer);
      if (inCenter) {
        centerSum += value;
        centerCount += 1;
      }
      if (inBorder) {
        borderSum += value;
        borderCount += 1;
      }
    }
  }

  const centerLuma = centerCount ? centerSum / centerCount : meanLuma;
  const borderLuma = borderCount ? borderSum / borderCount : meanLuma;
  const glowContrast = centerLuma - borderLuma;
  const screenGlow =
    blueDominance > BLUE_DOMINANCE_MAX ||
    (borderLuma < 70 && glowContrast > 55 && flatRatio > 0.2);

  const details = {
    mean_luma: Number(meanLuma.toFixed(2)),
    luma_std: Number(Math.sqrt(variance).toFixed(2)),
    strong_edges: strongEdges,
    axis_edge_ratio: Number(axisRatio.toFixed(4)),
    longest_row_span: Number(rowSpan.toFixed(4)),
    longest_column_span: Number(columnSpan.toFixed(4)),
    flat_block_ratio: Number(flatRatio.toFixed(4)),
    blue_dominance_ratio: Number(blueDominance.toFixed(4)),
    center_luma: Number(centerLuma.toFixed(2)),
    border_luma: Number(borderLuma.toFixed(2)),
    glow_contrast: Number(glowContrast.toFixed(2)),
  };

  if (axisRatio > AXIS_EDGE_RATIO_MAX || rowSpan > EDGE_SPAN_MAX || columnSpan > EDGE_SPAN_MAX) {
    return { rejected: true, reason_code: "screen_or_manmade_object", details };
  }
  if (screenGlow) {
    return { rejected: true, reason_code: "screen_or_manmade_object", details };
  }
  if (flatRatio > 0.55 || (flatRatio > FLAT_BLOCK_RATIO_MAX && strongEdges < 6000)) {
    return { rejected: true, reason_code: "unrelated_object", details };
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
