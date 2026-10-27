/**
 * Authoritative square-crop math + on-device quality gate.
 *
 * The model contract (src/data_pipeline.py::_eval_transform) is:
 *   shortest-side resize to 260 -> center crop 260 -> ImageNet normalize.
 * If the client sends an already-square crop, the server-side center crop is
 * the identity transform, so the region the user framed is the region the
 * model evaluates.
 *
 * All geometry helpers in this file are pure so they can be unit-tested in
 * Node (see __tests__/cropAndQuality.test.js). Canvas/DOM helpers are guarded
 * and only touched when called from the browser.
 */

export const MODEL_SIZE = 260;
export const EXPORT_SIZE = 520;
export const QUALITY_SAMPLE_SIZE = 130;
export const ZOOM_MIN = 1;
export const ZOOM_MAX = 4;

export const QUALITY_THRESHOLDS = {
  // 0-100 sharpness score below which a frame is considered blurry.
  blurScoreMin: 85,
  // Mean luminance bounds (0-255).
  darkMean: 45,
  brightMean: 225,
  // Fraction of crushed/blocked pixels.
  darkPixelPct: 0.5,
  brightPixelPct: 0.4,
  // No visible subject when the frame is near-black / crushed / washed out.
  noSubjectMean: 35,
  noSubjectDarkPct: 0.65,
  washedOutMean: 235,
  // Mean Sobel magnitude in the central 70% below which we assume no animal.
  minCenterEdge: 8,
  // Fraction of the outer ring with strong edges -> animal touches the crop.
  borderTouchPct: 0.28,
};

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/** Scale that makes a source image cover a viewport (CSS object-fit: cover). */
export function computeCoverScale(sourceW, sourceH, viewportW, viewportH) {
  if (!sourceW || !sourceH || !viewportW || !viewportH) return 1;
  return Math.max(viewportW / sourceW, viewportH / sourceH);
}

/**
 * Map the on-screen 1:1 target box to a source-pixel rectangle.
 *
 * @param {object} p
 * @param {number} p.viewportW  Stage width in CSS pixels.
 * @param {number} p.viewportH  Stage height in CSS pixels.
 * @param {number} p.sourceW    Raw video/image width in pixels.
 * @param {number} p.sourceH    Raw video/image height in pixels.
 * @param {number} p.reticleX    Target box left, relative to the stage.
 * @param {number} p.reticleY    Target box top, relative to the stage.
 * @param {number} p.reticleSize Target box side length in CSS pixels.
 * @param {number} [p.zoom]      User zoom, 1..ZOOM_MAX (scales about the stage centre).
 * @param {number} [p.panX]      User pan in CSS pixels (image moved right = +).
 * @param {number} [p.panY]      User pan in CSS pixels (image moved down = +).
 * @param {boolean} [p.mirrored] True when reading a raw front-camera buffer
 *                               that is displayed mirrored.
 * @returns {{x:number,y:number,size:number,scale:number,offsetX:number,offsetY:number}}
 */
export function computeSourceRect({
  viewportW,
  viewportH,
  sourceW,
  sourceH,
  reticleX,
  reticleY,
  reticleSize,
  zoom = 1,
  panX = 0,
  panY = 0,
  mirrored = false,
}) {
  const safeZoom = clamp(zoom || 1, ZOOM_MIN, ZOOM_MAX);
  // CSS applies `scale(zoom)` about the viewport centre on top of
  // `object-fit: cover`, so both the cover scale and the centring offset must
  // be evaluated at the zoomed scale. (Evaluating the offset at the base
  // scale would anchor zoom at the source origin instead of the viewport.)
  const scale = computeCoverScale(sourceW, sourceH, viewportW, viewportH) * safeZoom;

  // object-fit: cover centres the image in the viewport.
  const offsetX = (sourceW * scale - viewportW) / 2;
  const offsetY = (sourceH * scale - viewportH) / 2;

  const size = Math.min(
    Math.max(1, reticleSize / scale),
    Math.min(sourceW, sourceH)
  );

  let x = (reticleX + offsetX - panX) / scale;
  let y = (reticleY + offsetY - panY) / scale;

  x = clamp(x, 0, sourceW - size);
  y = clamp(y, 0, sourceH - size);

  if (mirrored) {
    x = sourceW - x - size;
  }

  return { x, y, size, scale, offsetX, offsetY };
}

/**
 * Keep the *target box* inside the source image.
 *
 * This is intentionally more permissive than clamping to the viewport: a
 * landscape photo in a 4:3 viewfinder has no vertical overflow, but the
 * smaller square target can still move up and down over the image. That is
 * what enables both-axis framing. The area outside the target is dimmed and
 * is never part of the model input.
 */
export function clampPan({
  panX,
  panY,
  zoom,
  viewportW,
  viewportH,
  sourceW,
  sourceH,
  reticleSize,
}) {
  const safeZoom = clamp(zoom || 1, ZOOM_MIN, ZOOM_MAX);
  const scale = computeCoverScale(sourceW, sourceH, viewportW, viewportH) * safeZoom;
  const target = Math.max(1, reticleSize || Math.min(viewportW, viewportH));
  const cropSize = Math.min(target / scale, Math.min(sourceW, sourceH));
  const maxX = Math.max(0, ((sourceW - cropSize) * scale) / 2);
  const maxY = Math.max(0, ((sourceH - cropSize) * scale) / 2);
  return {
    x: clamp(panX, -maxX, maxX),
    y: clamp(panY, -maxY, maxY),
  };
}

/** Shortest-side resize + center crop, mirroring the Python eval transform. */
export function centerSquareRect(sourceW, sourceH) {
  const side = Math.min(sourceW, sourceH);
  return {
    x: (sourceW - side) / 2,
    y: (sourceH - side) / 2,
    size: side,
  };
}

function toGrayscale(data, width, height) {
  const gray = new Float32Array(width * height);
  let sum = 0;
  let dark = 0;
  let bright = 0;
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const y = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    gray[i] = y;
    sum += y;
    if (y < 25) dark += 1;
    if (y > 245) bright += 1;
  }
  return {
    gray,
    mean: sum / gray.length,
    darkPct: dark / gray.length,
    brightPct: bright / gray.length,
  };
}

function sobelMagnitude(gray, width, height, x, y) {
  const i = y * width + x;
  const tl = gray[i - width - 1];
  const t = gray[i - width];
  const tr = gray[i - width + 1];
  const l = gray[i - 1];
  const r = gray[i + 1];
  const bl = gray[i + width - 1];
  const b = gray[i + width];
  const br = gray[i + width + 1];
  const gx = -tl - 2 * l - bl + tr + 2 * r + br;
  const gy = -tl - 2 * t - tr + bl + 2 * b + br;
  return Math.hypot(gx, gy);
}

/**
 * Score a 130x130-ish RGBA buffer.
 * @returns {{overallScore:number,sharpnessScore:number,exposureScore:number,
 *   framingScore:number,meanLuma:number,darkPct:number,brightPct:number,
 *   laplacianVariance:number,centerEdge:number,borderTouch:number,
 *   lightingState:string,framingState:string,sharpnessState:string,
 *   noSubject:boolean,partialOut:boolean,status:string,flags:string[],isReady:boolean}}
 */
export function analyzeImageData({ data, width, height }) {
  const { gray, mean, darkPct, brightPct } = toGrayscale(data, width, height);

  // 4-neighbour Laplacian variance on the interior.
  let lapSum = 0;
  let lapSqSum = 0;
  let lapCount = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - width] - gray[i + width];
      lapSum += lap;
      lapSqSum += lap * lap;
      lapCount += 1;
    }
  }
  const lapMean = lapCount ? lapSum / lapCount : 0;
  const laplacianVariance = lapCount ? lapSqSum / lapCount - lapMean * lapMean : 0;

  // Smooth 0-100 curve: flat frame -> 0, textured frame -> 80-100.
  const sharpnessScore = clamp(
    Math.round(100 * (1 - Math.exp(-Math.max(0, laplacianVariance) / 260))),
    0,
    100
  );

  // Central 70% vs outer ring edge energy.
  const cx0 = Math.floor(width * 0.15);
  const cx1 = Math.ceil(width * 0.85);
  const cy0 = Math.floor(height * 0.15);
  const cy1 = Math.ceil(height * 0.85);
  let centerSum = 0;
  let centerCount = 0;
  let borderSum = 0;
  let borderCount = 0;
  let borderStrong = 0;

  const edge = (x, y) => sobelMagnitude(gray, width, height, x, y);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const mag = edge(x, y);
      const inCenter = x >= cx0 && x < cx1 && y >= cy0 && y < cy1;
      const inBorderRing =
        x < Math.floor(width * 0.1) ||
        x >= Math.ceil(width * 0.9) ||
        y < Math.floor(height * 0.1) ||
        y >= Math.ceil(height * 0.9);
      if (inCenter) {
        centerSum += mag;
        centerCount += 1;
      }
      if (inBorderRing) {
        borderSum += mag;
        borderCount += 1;
        if (mag > 60) borderStrong += 1;
      }
    }
  }
  const centerEdge = centerCount ? centerSum / centerCount : 0;
  const borderEdge = borderCount ? borderSum / borderCount : 0;
  const borderTouch = borderCount ? borderStrong / borderCount : 0;

  const exposureScore = clamp(
    Math.round(100 - Math.abs(mean - 128) * 0.55 - darkPct * 60 - brightPct * 60),
    0,
    100
  );
  let framingScore = clamp(
    Math.round(centerEdge * 4 - borderTouch * 110),
    0,
    100
  );

  const flags = [];
  if (sharpnessScore < QUALITY_THRESHOLDS.blurScoreMin) flags.push("TOO_BLURRY");
  if (mean < QUALITY_THRESHOLDS.darkMean || darkPct > QUALITY_THRESHOLDS.darkPixelPct) {
    flags.push("TOO_DARK");
  }
  if (mean > QUALITY_THRESHOLDS.brightMean || brightPct > QUALITY_THRESHOLDS.brightPixelPct) {
    flags.push("OVEREXPOSED");
  }

  // A dark/crushed/washed-out frame with no central edge energy cannot contain
  // a visible animal. This is deliberately stricter than the old
  // centerEdge < 6 check, which let pitch-black frames report "framing ready".
  const noSubject =
    mean < QUALITY_THRESHOLDS.noSubjectMean ||
    darkPct > QUALITY_THRESHOLDS.noSubjectDarkPct ||
    mean > QUALITY_THRESHOLDS.washedOutMean ||
    centerEdge < QUALITY_THRESHOLDS.minCenterEdge;
  const partialOut =
    !noSubject &&
    borderTouch > QUALITY_THRESHOLDS.borderTouchPct &&
    borderEdge >= centerEdge;

  if (noSubject) {
    flags.push("NO_SUBJECT_OR_TOO_FAR");
    framingScore = Math.min(framingScore, 15);
  } else if (partialOut) {
    flags.push("PARTIALLY_OUT_OF_FRAME");
    // Keep the score coherent: a frame cannot be "touching the edge" and
    // simultaneously report perfect framing.
    framingScore = Math.min(framingScore, 55);
  }

  const lightingState =
    mean < QUALITY_THRESHOLDS.darkMean || darkPct > QUALITY_THRESHOLDS.darkPixelPct
      ? "too_dark"
      : mean > QUALITY_THRESHOLDS.brightMean || brightPct > QUALITY_THRESHOLDS.brightPixelPct
      ? "overexposed"
      : "good";

  const framingState =
    !noSubject && !partialOut && mean >= QUALITY_THRESHOLDS.noSubjectMean && centerEdge >= QUALITY_THRESHOLDS.minCenterEdge
      ? "ready"
      : noSubject || mean < QUALITY_THRESHOLDS.noSubjectMean || centerEdge < QUALITY_THRESHOLDS.minCenterEdge
      ? "no_subject"
      : "edge_clipped";

  const sharpnessState =
    sharpnessScore >= QUALITY_THRESHOLDS.blurScoreMin && mean >= QUALITY_THRESHOLDS.noSubjectMean
      ? "optimal"
      : "blurry";

  const status = flags[0] || "READY";
  const overallScore = clamp(
    Math.round(sharpnessScore * 0.45 + exposureScore * 0.35 + framingScore * 0.2),
    0,
    100
  );

  return {
    overallScore,
    sharpnessScore,
    exposureScore,
    framingScore,
    meanLuma: Math.round(mean),
    darkPct,
    brightPct,
    laplacianVariance: Math.round(laplacianVariance),
    centerEdge: Number(centerEdge.toFixed(2)),
    borderTouch: Number(borderTouch.toFixed(3)),
    lightingState,
    framingState,
    sharpnessState,
    noSubject,
    partialOut,
    status,
    flags,
    isReady: status === "READY",
  };
}

/** Draw a source-pixel rectangle into a square canvas. */
export function drawCropToCanvas(source, rect, size = EXPORT_SIZE) {
  if (typeof document === "undefined") {
    throw new Error("drawCropToCanvas requires a browser environment");
  }
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(
    source,
    Math.round(rect.x),
    Math.round(rect.y),
    Math.round(rect.size),
    Math.round(rect.size),
    0,
    0,
    size,
    size
  );
  return canvas;
}

/** Draw an arbitrary source rect into an offscreen quality-analysis canvas. */
export function sampleCropForQuality(source, rect, size = QUALITY_SAMPLE_SIZE) {
  return drawCropToCanvas(source, rect, size);
}

export function analyzeSourceCrop(source, rect) {
  const canvas = sampleCropForQuality(source, rect);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  return analyzeImageData(ctx.getImageData(0, 0, canvas.width, canvas.height));
}

/** Explicit quality-gate entry point used by uploads and the live camera. */
export function analyzeCanvasQuality(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  return analyzeImageData(ctx.getImageData(0, 0, canvas.width, canvas.height));
}

export function analyzeCanvas(canvas) {
  return analyzeCanvasQuality(canvas);
}

/**
 * Render any image data URL into a 260×260 canvas and score it with the same
 * heuristic used by the live camera. This is the authoritative quality report
 * for uploads and manual crops (whose intermediate crop is 520×520).
 */
export async function analyzeDataUrlQuality(dataUrl, size = MODEL_SIZE) {
  if (!dataUrl || typeof document === "undefined") return null;
  const image = await loadImage(dataUrl);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  const scale = Math.max(size / image.width, size / image.height);
  const sw = size / scale;
  const sh = size / scale;
  ctx.drawImage(image, (image.width - sw) / 2, (image.height - sh) / 2, sw, sh, 0, 0, size, size);
  const report = analyzeCanvasQuality(canvas);
  return { ...report, sampleSize: size };
}

export function canvasToBlob(canvas, type = "image/jpeg", quality = 0.92) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Canvas encoding failed"))),
      type,
      quality
    );
  });
}

export async function makeCroppedFile(
  source,
  rect,
  { size = EXPORT_SIZE, filename = "capture.jpg", type = "image/jpeg", quality = 0.92 } = {}
) {
  const canvas = drawCropToCanvas(source, rect, size);
  const blob = await canvasToBlob(canvas, type, quality);
  const dataUrl = canvas.toDataURL(type, quality);
  const file = new File([blob], filename, { type });
  return { file, blob, dataUrl, canvas };
}

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not load image"));
    img.src = url;
  });
}

export function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Could not read file"));
    reader.readAsDataURL(file);
  });
}

export function captureVideoFrame(video, mirrored = false) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (mirrored) {
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

// ---------------------------------------------------------------------------
// Shareable result card (PNG)
// ---------------------------------------------------------------------------

export function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function drawCover(ctx, image, x, y, w, h) {
  const scale = Math.max(w / image.width, h / image.height);
  const sw = w / scale;
  const sh = h / scale;
  const sx = (image.width - sw) / 2;
  const sy = (image.height - sh) / 2;
  ctx.drawImage(image, sx, sy, sw, sh, x, y, w, h);
}

function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
  const value = String(text || "");
  if (ctx.measureText(value).width <= maxWidth) {
    ctx.fillText(value, x, y);
    return;
  }
  let line = "";
  for (const char of value) {
    const test = line + char;
    if (ctx.measureText(`${test}…`).width > maxWidth && line) {
      ctx.fillText(`${line}…`, x, y);
      return;
    }
    line = test;
  }
  ctx.fillText(line, x, y);
}

/**
 * Render a branded 1080x1350 result card.
 * @param {HTMLCanvasElement} canvas
 * @param {object} opts {image, result, quality, traits}
 *   `image` must be a preloaded HTMLImageElement (canvas drawImage is sync).
 */
export function renderShareCard(canvas, { image, result, quality, traits } = {}) {
  const W = 1080;
  const H = 1350;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");

  const bg = ctx.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, "#0b0f19");
  bg.addColorStop(1, "#111827");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  const accent = ctx.createLinearGradient(0, 0, W, 0);
  accent.addColorStop(0, "#6366f1");
  accent.addColorStop(1, "#10b981");
  ctx.fillStyle = accent;
  ctx.fillRect(0, 0, W, 10);

  ctx.fillStyle = "#f1f5f9";
  ctx.font = "700 34px Inter, system-ui, sans-serif";
  ctx.fillText("BreedID Livestock Vision", 64, 78);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "500 22px Inter, system-ui, sans-serif";
  ctx.fillText("Cattle & Buffalo Breed Identification", 64, 112);

  const photoX = 64;
  const photoY = 150;
  const photoW = W - 128;
  const photoH = 640;

  ctx.save();
  roundRect(ctx, photoX, photoY, photoW, photoH, 28);
  ctx.clip();
  ctx.fillStyle = "#1e293b";
  ctx.fillRect(photoX, photoY, photoW, photoH);
  if (image && typeof HTMLImageElement !== "undefined" && image instanceof HTMLImageElement) {
    drawCover(ctx, image, photoX, photoY, photoW, photoH);
  }
  ctx.restore();

  ctx.save();
  roundRect(ctx, photoX, photoY + photoH - 74, 300, 54, 27);
  ctx.fillStyle = "rgba(11, 15, 25, 0.82)";
  ctx.fill();
  ctx.fillStyle = "#34d399";
  ctx.font = "700 24px Inter, system-ui, sans-serif";
  ctx.fillText((result?.predicted_species || result?.species || "Cattle").toUpperCase(), photoX + 24, photoY + photoH - 38);
  ctx.restore();

  let y = 850;
  ctx.fillStyle = "#f1f5f9";
  ctx.font = "800 54px Inter, system-ui, sans-serif";
  const breed = (result?.predicted_breed || result?.breed || "Unknown breed").replace(/_/g, " ");
  ctx.fillText(breed, 64, y);

  const confidence = Math.round((result?.confidence ?? 0) * 100);
  y += 56;
  ctx.fillStyle = "#94a3b8";
  ctx.font = "500 24px Inter, system-ui, sans-serif";
  ctx.fillText(`${confidence}% breed confidence`, 64, y);

  if (quality?.overallScore != null) {
    ctx.fillStyle = "#34d399";
    ctx.font = "700 24px Inter, system-ui, sans-serif";
    ctx.fillText(`Image quality ${quality.overallScore}/100`, 620, y);
  }

  // Top-3 bars
  const top = (result?.top_k || result?.top5_breeds || []).slice(0, 3);
  y += 46;
  const barW = W - 128;
  top.forEach((item, idx) => {
    const prob = typeof item.prob === "number" ? item.prob : (item.confidence || 0) / 100;
    const name = (item.breed || "").replace(/_/g, " ");
    ctx.fillStyle = idx === 0 ? "#e2e8f0" : "#94a3b8";
    ctx.font = `${idx === 0 ? "700" : "500"} 24px Inter, system-ui, sans-serif`;
    ctx.fillText(`${idx + 1}. ${name}`, 64, y);
    ctx.textAlign = "right";
    ctx.fillText(`${Math.round(prob * 100)}%`, W - 64, y);
    ctx.textAlign = "left";
    y += 16;
    ctx.fillStyle = "rgba(148, 163, 184, 0.18)";
    roundRect(ctx, 64, y, barW, 14, 7);
    ctx.fill();
    ctx.fillStyle = idx === 0 ? "#6366f1" : "#475569";
    roundRect(ctx, 64, y, Math.max(8, barW * prob), 14, 7);
    ctx.fill();
    y += 40;
  });

  if (traits) {
    y += 8;
    ctx.fillStyle = "#94a3b8";
    ctx.font = "600 20px Inter, system-ui, sans-serif";
    ctx.fillText("KEY MORPHOLOGY", 64, y);
    y += 34;
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "500 24px Inter, system-ui, sans-serif";
    const lines = [
      `Hump: ${traits.hump || "—"}`,
      `Horns: ${traits.horn || "—"}`,
      `Dewlap: ${traits.dewlap || "—"}`,
    ];
    lines.forEach((line) => {
      ctx.fillText(line, 64, y);
      y += 34;
    });
  }

  ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
  ctx.beginPath();
  ctx.moveTo(64, 1256);
  ctx.lineTo(W - 64, 1256);
  ctx.stroke();

  ctx.fillStyle = "#94a3b8";
  ctx.font = "500 20px Inter, system-ui, sans-serif";
  ctx.fillText(`Model: ${result?.model_name || result?.model_used || "unknown"}`, 64, 1290);
  wrapText(ctx, `File: ${result?.filename || "unknown"}`, 64, 1322, W - 360, 20);
  ctx.textAlign = "right";
  ctx.fillStyle = "#6366f1";
  ctx.font = "700 20px Inter, system-ui, sans-serif";
  ctx.fillText("Soft-routed 3-head EfficientNet-Lite", W - 64, 1290);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "500 20px Inter, system-ui, sans-serif";
  ctx.fillText(new Date().toLocaleString(), W - 64, 1322);
  ctx.textAlign = "left";

  return canvas;
}

export async function shareOrDownloadCanvas(canvas, filename = "breed-result.png") {
  const blob = await canvasToBlob(canvas, "image/png", 1);
  const file = new File([blob], filename, { type: "image/png" });
  let canShare = false;
  try {
    canShare = Boolean(typeof navigator !== "undefined" && navigator.canShare?.({ files: [file] }));
  } catch {
    canShare = false;
  }
  if (canShare) {
    try {
      await navigator.share({ files: [file], title: "Breed identification" });
      return "shared";
    } catch (err) {
      if (err?.name === "AbortError") return "cancelled";
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  return "downloaded";
}
