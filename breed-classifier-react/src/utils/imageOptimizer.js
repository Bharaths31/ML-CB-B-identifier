/**
 * Client-side 260x260 optimizer.
 *
 * The server contract is `Resize(260) -> CenterCrop(260)` (src/data_pipeline.py).
 * If the client sends a 260x260 image, both server steps are identity
 * transforms. This module builds that image without cropping the animal:
 *
 *   pad    : scale the longest side to 260, centre the result, fill the rest
 *            with the ImageNet mean (which is ~0 after normalization).
 *   stretch: resize directly to 260x260 (no bars, no crop).
 *
 * Progressive halving avoids the aliasing you get when a 12 MP photo is drawn
 * straight onto a 260px canvas.
 */

export const TARGET_SIZE = 260;
export const IMAGENET_MEAN_RGB = [124, 116, 104];
export const IMAGENET_MEAN_HEX = "#7c7468";
export const FIT_MODES = ["pad", "stretch"];
export const PAD_STYLES = ["imagenet_mean", "ambient_blur", "black"];

const DEFAULT_OPTIONS = {
  fitMode: "pad",
  padStyle: "imagenet_mean",
  allowUpscale: false,
  mimeType: "image/jpeg",
  quality: 0.9,
};

const now = () =>
  typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();

function gcd(a, b) {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y) {
    [x, y] = [y, x % y];
  }
  return x || 1;
}

const COMMON_RATIOS = [
  [1, 1],
  [4, 3],
  [3, 4],
  [16, 9],
  [9, 16],
  [3, 2],
  [2, 3],
  [5, 4],
  [4, 5],
  [21, 9],
  [9, 21],
];

export function aspectRatio(width, height) {
  if (!width || !height) return "—";
  const ratio = width / height;
  // Phone images are often resized to odd dimensions (e.g. 1280x853), so
  // reduce to the nearest common ratio before falling back to exact GCD.
  const common = COMMON_RATIOS.find(
    ([w, h]) => Math.abs(ratio - w / h) / (w / h) < 0.02
  );
  if (common) return `${common[0]}:${common[1]}`;
  const g = gcd(width, height);
  return `${Math.round(width / g)}:${Math.round(height / g)}`;
}

export function formatBytes(bytes) {
  if (bytes == null || Number.isNaN(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function normalizeOptions(options = {}) {
  const fitMode = FIT_MODES.includes(options.fitMode) ? options.fitMode : DEFAULT_OPTIONS.fitMode;
  const padStyle = PAD_STYLES.includes(options.padStyle) ? options.padStyle : DEFAULT_OPTIONS.padStyle;
  return {
    ...DEFAULT_OPTIONS,
    ...options,
    fitMode,
    padStyle,
    allowUpscale: Boolean(options.allowUpscale),
  };
}

/**
 * Pure layout math for the 260x260 canvas.
 * @returns {{targetWidth:number,targetHeight:number,innerWidth:number,
 *   innerHeight:number,padX:number,padY:number,scaleFactor:number,
 *   scaleX:number,scaleY:number,upscaled:boolean,fitMode:string}}
 */
export function computeOptimizerLayout(width, height, options = {}) {
  if (!width || !height) {
    throw new Error("computeOptimizerLayout requires positive width and height");
  }
  const { fitMode, allowUpscale } = normalizeOptions(options);

  if (fitMode === "stretch") {
    const scaleX = TARGET_SIZE / width;
    const scaleY = TARGET_SIZE / height;
    return {
      targetWidth: TARGET_SIZE,
      targetHeight: TARGET_SIZE,
      innerWidth: TARGET_SIZE,
      innerHeight: TARGET_SIZE,
      padX: 0,
      padY: 0,
      scaleFactor: TARGET_SIZE / Math.max(width, height),
      scaleX,
      scaleY,
      upscaled: scaleX > 1 || scaleY > 1,
      fitMode,
    };
  }

  const maxSide = Math.max(width, height);
  // The product requirement is "only downscale": default to no upscaling.
  // Small images are padded rather than enlarged unless explicitly requested.
  const scaleFactor = allowUpscale ? TARGET_SIZE / maxSide : Math.min(1, TARGET_SIZE / maxSide);
  const innerWidth = Math.max(1, Math.round(width * scaleFactor));
  const innerHeight = Math.max(1, Math.round(height * scaleFactor));
  const padX = Math.floor((TARGET_SIZE - innerWidth) / 2);
  const padY = Math.floor((TARGET_SIZE - innerHeight) / 2);

  return {
    targetWidth: TARGET_SIZE,
    targetHeight: TARGET_SIZE,
    innerWidth,
    innerHeight,
    padX,
    padY,
    scaleFactor,
    scaleX: scaleFactor,
    scaleY: scaleFactor,
    upscaled: scaleFactor > 1,
    fitMode,
  };
}

function createCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  return canvas;
}

/** Progressive halving, then a final high-quality draw into `target`. */
export function stepDownResize(source, srcW, srcH, targetW, targetH) {
  let current = source;
  let curW = srcW;
  let curH = srcH;

  while (curW * 0.5 > targetW && curH * 0.5 > targetH) {
    const nextW = Math.max(1, Math.round(curW / 2));
    const nextH = Math.max(1, Math.round(curH / 2));
    const staging = createCanvas(nextW, nextH);
    const ctx = staging.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(current, 0, 0, curW, curH, 0, 0, nextW, nextH);
    current = staging;
    curW = nextW;
    curH = nextH;
  }

  return { source: current, width: curW, height: curH };
}

function fillPad(ctx, source, sourceWidth, sourceHeight, padStyle) {
  if (padStyle === "black") {
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
    return;
  }

  if (padStyle === "ambient_blur") {
    ctx.save();
    // Stretch a blurred, darkened copy behind the sharp letterboxed image.
    ctx.filter = "blur(12px) brightness(0.6)";
    ctx.drawImage(source, 0, 0, sourceWidth, sourceHeight, 0, 0, TARGET_SIZE, TARGET_SIZE);
    ctx.restore();
    return;
  }

  // ImageNet mean: matches src/data_pipeline.py::_PadToSquare, ~0 after
  // Normalize(mean=[0.485,0.456,0.406]).
  ctx.fillStyle = IMAGENET_MEAN_HEX;
  ctx.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
}

async function decodeWithCreateImageBitmap(blob) {
  if (typeof createImageBitmap !== "function") return null;
  try {
    const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
    return {
      source: bitmap,
      width: bitmap.width,
      height: bitmap.height,
      close: () => bitmap.close?.(),
    };
  } catch {
    return null;
  }
}

function decodeWithImage(blob) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () =>
      resolve({
        source: img,
        width: img.naturalWidth,
        height: img.naturalHeight,
        close: () => URL.revokeObjectURL(url),
      });
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not decode image"));
    };
    img.src = url;
  });
}

async function decodeImage(blob) {
  return (await decodeWithCreateImageBitmap(blob)) || decodeWithImage(blob);
}

function canvasToBlob(canvas, mimeType, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Canvas encoding failed"))),
      mimeType,
      quality
    );
  });
}

function safeFileName(name, mimeType) {
  const base = (name || "image").replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "_");
  const ext = mimeType === "image/png" ? "png" : "jpg";
  return `zero_crop_260_${base}.${ext}`;
}

/**
 * Optimize an already-decoded source (canvas, image, ImageBitmap) to 260x260.
 * Used by the camera, where the source is a captured canvas.
 */
export async function optimizeSourceTo260(source, sourceWidth, sourceHeight, options = {}) {
  const opts = normalizeOptions(options);
  const started = options.startTime ?? now();
  const layout = computeOptimizerLayout(sourceWidth, sourceHeight, opts);

  const canvas = createCanvas(TARGET_SIZE, TARGET_SIZE);
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  if (layout.fitMode === "pad") {
    fillPad(ctx, source, sourceWidth, sourceHeight, opts.padStyle);
  }

  const targetW = layout.fitMode === "stretch" ? TARGET_SIZE : layout.innerWidth;
  const targetH = layout.fitMode === "stretch" ? TARGET_SIZE : layout.innerHeight;
  const staged = stepDownResize(source, sourceWidth, sourceHeight, targetW, targetH);
  ctx.drawImage(
    staged.source,
    0,
    0,
    staged.width,
    staged.height,
    layout.padX,
    layout.padY,
    targetW,
    targetH
  );

  const blob = await canvasToBlob(canvas, opts.mimeType, opts.quality);
  const dataUrl = canvas.toDataURL(opts.mimeType, opts.quality);
  const file = new File([blob], safeFileName(options.originalName, opts.mimeType), {
    type: opts.mimeType,
  });
  const originalBytes = Number(options.originalBytes) || 0;
  const optimizedBytes = blob.size;
  const durationMs = Math.max(0, now() - started);

  return {
    file,
    blob,
    dataUrl,
    originalDataUrl: options.originalDataUrl || null,
    meta: {
      originalWidth: sourceWidth,
      originalHeight: sourceHeight,
      originalAspectRatio: aspectRatio(sourceWidth, sourceHeight),
      originalBytes,
      optimizedWidth: TARGET_SIZE,
      optimizedHeight: TARGET_SIZE,
      scaledInnerWidth: layout.innerWidth,
      scaledInnerHeight: layout.innerHeight,
      padX: layout.padX,
      padY: layout.padY,
      scaleFactor: layout.scaleFactor,
      scaleX: layout.scaleX,
      scaleY: layout.scaleY,
      upscaled: layout.upscaled,
      fitMode: layout.fitMode,
      padStyle: opts.padStyle,
      optimizedBytes,
      savingsPct:
        originalBytes > 0
          ? Math.round((1 - optimizedBytes / originalBytes) * 1000) / 10
          : 0,
      durationMs: Math.round(durationMs * 10) / 10,
    },
  };
}

/**
 * Decode `fileOrBlob` (EXIF-upright when supported), optimize it, and return
 * the File/Blob/dataUrl/meta contract consumed by the UI.
 */
export async function optimizeImageTo260(fileOrBlob, options = {}) {
  const started = now();
  const decoded = await decodeImage(fileOrBlob);
  try {
    const originalDataUrl =
      options.originalDataUrl ||
      (typeof URL !== "undefined" && URL.createObjectURL
        ? URL.createObjectURL(fileOrBlob)
        : null);
    return await optimizeSourceTo260(decoded.source, decoded.width, decoded.height, {
      ...options,
      startTime: started,
      originalName: options.originalName || fileOrBlob.name || "image.jpg",
      originalBytes: options.originalBytes ?? fileOrBlob.size ?? 0,
      originalDataUrl,
    });
  } finally {
    decoded.close?.();
  }
}

/** Revoke the object URL created for the original preview, when present. */
export function releaseOptimizerResult(result) {
  const url = result?.originalDataUrl;
  if (url && typeof url === "string" && url.startsWith("blob:") && typeof URL !== "undefined") {
    URL.revokeObjectURL(url);
  }
}
