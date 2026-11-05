/**
 * Upload/camera image preparation for the standalone Android app.
 *
 * Uploaded gallery files expose three crop modes and two downscale modes.
 * Camera captures always use the default `center` + `auto` pipeline with no
 * manual UI, matching the server's Resize(260) + CenterCrop(260) behaviour.
 */

import { IMAGE_SIZE, IMAGENET_MEAN } from "./breedCatalog.js";

const PAD_FILL = `rgb(${Math.round(IMAGENET_MEAN[0] * 255)}, ${Math.round(IMAGENET_MEAN[1] * 255)}, ${Math.round(IMAGENET_MEAN[2] * 255)})`;

export const CROP_MODES = [
  { id: "pad", label: "Pad to Square (Full Body)" },
  { id: "center", label: "Center Crop (260×260)" },
  { id: "manual", label: "Manual Drag Crop" },
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

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not decode image"));
    img.src = src;
  });
}

/**
 * Decode an image file without round-tripping it through base64.
 *
 * Gallery photos are often several megabytes; reading them as a data URL and
 * decoding that string twice is a large, main-thread-visible delay on phones.
 * `createImageBitmap` decodes off the main thread (and applies EXIF rotation),
 * with an object-URL image element as the fallback for older WebViews.
 */
export async function loadImageFile(file) {
  if (!file) throw new Error("No image file provided.");
  const objectUrl = URL.createObjectURL(file);
  let image;
  let release;
  if (typeof createImageBitmap === "function") {
    try {
      image = await createImageBitmap(file, { imageOrientation: "from-image" });
      release = () => image.close?.();
      URL.revokeObjectURL(objectUrl);
    } catch {
      image = await loadImageElement(objectUrl);
      release = () => URL.revokeObjectURL(objectUrl);
    }
  } else {
    image = await loadImageElement(objectUrl);
    release = () => URL.revokeObjectURL(objectUrl);
  }
  return {
    image,
    width: image.naturalWidth || image.width,
    height: image.naturalHeight || image.height,
    release,
  };
}

function drawPrepared(image, cropMode) {
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  const w = image.naturalWidth || image.width;
  const h = image.naturalHeight || image.height;
  if (!w || !h) throw new Error("Image has no dimensions.");

  if (cropMode === "pad") {
    const scale = IMAGE_SIZE / Math.max(w, h);
    const dw = Math.max(1, Math.round(w * scale));
    const dh = Math.max(1, Math.round(h * scale));
    ctx.fillStyle = PAD_FILL;
    ctx.fillRect(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    ctx.drawImage(image, Math.round((IMAGE_SIZE - dw) / 2), Math.round((IMAGE_SIZE - dh) / 2), dw, dh);
  } else {
    const scale = IMAGE_SIZE / Math.min(w, h);
    const dw = Math.max(1, Math.round(w * scale));
    const dh = Math.max(1, Math.round(h * scale));
    ctx.drawImage(image, Math.round((IMAGE_SIZE - dw) / 2), Math.round((IMAGE_SIZE - dh) / 2), dw, dh);
  }
  return canvas;
}

/**
 * Prepare a gallery upload or a manual crop.
 *
 * @returns {Promise<{file:File,dataUrl:string,canvas:HTMLCanvasElement,meta:object}>}
 */
export async function prepareImage(file, {
  cropMode = "pad",
  manualCropFile = null,
} = {}) {
  const sourceFile = manualCropFile || file;
  const loaded = await loadImageFile(sourceFile);
  try {
    const effectiveCrop = manualCropFile ? "center" : cropMode;
    const canvas = drawPrepared(loaded.image, effectiveCrop);
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
        directDownscale: true,
        originalWidth: loaded.width,
        originalHeight: loaded.height,
        originalAspect: loaded.width / Math.max(1, loaded.height),
        outputWidth: IMAGE_SIZE,
        outputHeight: IMAGE_SIZE,
        source: "upload",
      },
    };
  } finally {
    loaded.release?.();
  }
}

/** Camera captures use the same direct center-crop path. */
export async function prepareCameraImage(file) {
  return prepareImage(file, { cropMode: "center" });
}
