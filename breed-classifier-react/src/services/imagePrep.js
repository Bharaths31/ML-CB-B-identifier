/**
 * Upload/camera image preparation for the standalone Android app.
 *
 * Uploaded gallery files expose three crop modes and two downscale modes.
 * Camera captures always use the default `center` + `auto` pipeline with no
 * manual UI, matching the server's Resize(260) + CenterCrop(260) behaviour.
 */

import { IMAGE_SIZE, IMAGENET_MEAN, TRAIN_RESIZE } from "./breedCatalog.js";

const PAD_FILL = `rgb(${Math.round(IMAGENET_MEAN[0] * 255)}, ${Math.round(IMAGENET_MEAN[1] * 255)}, ${Math.round(IMAGENET_MEAN[2] * 255)})`;

export const CROP_MODES = [
  { id: "pad", label: "Pad to Square (Full Body)" },
  { id: "center", label: "Center Crop (260×260)" },
  { id: "manual", label: "Manual Drag Crop" },
];

export const SCALE_MODES = [
  { id: "auto", label: "Auto (Shortest-side 260px)" },
  { id: "train", label: "Train Match (288px → 260px)" },
];

function canvasToBlob(canvas, mimeType = "image/jpeg", quality = 0.92) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not encode prepared image"))),
      mimeType,
      quality,
    );
  });
}

function canvasToDataUrl(canvas) {
  return canvas.toDataURL("image/jpeg", 0.92);
}

export async function loadImageFile(file) {
  if (!file) throw new Error("No image file provided.");
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Could not read image"));
    reader.readAsDataURL(file);
  });
  const image = await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not decode image"));
    img.src = dataUrl;
  });
  return { image, dataUrl, width: image.naturalWidth || image.width, height: image.naturalHeight || image.height };
}

function drawPrepared(image, cropMode, base) {
  const canvas = document.createElement("canvas");
  canvas.width = base;
  canvas.height = base;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  const w = image.naturalWidth || image.width;
  const h = image.naturalHeight || image.height;
  if (!w || !h) throw new Error("Image has no dimensions.");

  if (cropMode === "pad") {
    const scale = base / Math.max(w, h);
    const dw = Math.max(1, Math.round(w * scale));
    const dh = Math.max(1, Math.round(h * scale));
    ctx.fillStyle = PAD_FILL;
    ctx.fillRect(0, 0, base, base);
    ctx.drawImage(image, Math.round((base - dw) / 2), Math.round((base - dh) / 2), dw, dh);
  } else {
    // center / manual: cover the square and clip the overflow.
    const scale = base / Math.min(w, h);
    const dw = Math.max(1, Math.round(w * scale));
    const dh = Math.max(1, Math.round(h * scale));
    ctx.drawImage(image, Math.round((base - dw) / 2), Math.round((base - dh) / 2), dw, dh);
  }
  return canvas;
}

function centerCrop260(baseCanvas) {
  if (baseCanvas.width === IMAGE_SIZE && baseCanvas.height === IMAGE_SIZE) {
    return baseCanvas;
  }
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const offset = Math.round((baseCanvas.width - IMAGE_SIZE) / 2);
  ctx.drawImage(baseCanvas, offset, offset, IMAGE_SIZE, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  return canvas;
}

/**
 * Prepare a gallery upload or a manual crop.
 *
 * @returns {Promise<{file:File,dataUrl:string,canvas:HTMLCanvasElement,meta:object}>}
 */
export async function prepareImage(file, {
  cropMode = "pad",
  scaleMode = "auto",
  manualCropFile = null,
} = {}) {
  const sourceFile = manualCropFile || file;
  const loaded = await loadImageFile(sourceFile);
  const base = scaleMode === "train" ? TRAIN_RESIZE : IMAGE_SIZE;
  const effectiveCrop = manualCropFile ? "center" : cropMode;
  const baseCanvas = drawPrepared(loaded.image, effectiveCrop, base);
  const canvas = centerCrop260(baseCanvas);
  const blob = await canvasToBlob(canvas);
  const preparedFile = new File(
    [blob],
    `prepared_${Date.now()}.jpg`,
    { type: "image/jpeg", lastModified: Date.now() },
  );
  return {
    file: preparedFile,
    dataUrl: canvasToDataUrl(canvas),
    canvas,
    meta: {
      cropMode: effectiveCrop,
      scaleMode,
      baseSize: base,
      originalWidth: loaded.width,
      originalHeight: loaded.height,
      originalAspect: loaded.width / Math.max(1, loaded.height),
      outputWidth: IMAGE_SIZE,
      outputHeight: IMAGE_SIZE,
      source: "upload",
    },
  };
}

/** Camera captures never expose crop/downscale controls. */
export async function prepareCameraImage(file) {
  return prepareImage(file, { cropMode: "center", scaleMode: "auto" });
}
