import { useCallback, useEffect, useRef, useState } from "react";
import { fileToDataUrl } from "../utils/cropAndQuality.js";
import {
  optimizeImageTo260,
  releaseOptimizerResult,
} from "../utils/imageOptimizer.js";

export const DEFAULT_FIT = { fitMode: "stretch", padStyle: "imagenet_mean" };

/**
 * Owns the upload's original image, its current 260x260 optimized result, and
 * the fit configuration. Keeping this above `ImageUpload` lets the header
 * Advanced panel control the same optimizer instance without duplicating state.
 *
 * `original` never changes when the user applies a manual crop; the cropper
 * always re-opens the original upload, not the previous crop.
 */
export function useImageOptimizer(initialFit = DEFAULT_FIT) {
  const [original, setOriginal] = useState(null); // { file, dataUrl, name, bytes }
  const [cropped, setCropped] = useState(null); // { file, dataUrl }
  const [result, setResult] = useState(null);
  const [fit, setFit] = useState(initialFit);
  const [clahe, setClahe] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const requestRef = useRef(0);
  const resultRef = useRef(null);
  const claheRef = useRef(false);

  useEffect(() => {
    claheRef.current = clahe;
  }, [clahe]);

  const release = useCallback(() => {
    if (resultRef.current) {
      releaseOptimizerResult(resultRef.current);
      resultRef.current = null;
    }
  }, []);

  useEffect(
    () => () => {
      requestRef.current += 1;
      release();
    },
    [release]
  );

  const run = useCallback(
    async (sourceFile, nextFit, { croppedSource = false } = {}) => {
      const requestId = ++requestRef.current;
      setBusy(true);
      setError(null);
      try {
        const next = await optimizeImageTo260(sourceFile, {
          ...nextFit,
          clahe: claheRef.current,
          allowUpscale: false,
        });
        if (requestId !== requestRef.current) {
          releaseOptimizerResult(next);
          return null;
        }
        if (croppedSource) next.meta.manuallyCropped = true;
        release();
        resultRef.current = next;
        setResult(next);
        setBusy(false);
        return next;
      } catch (err) {
        if (requestId === requestRef.current) {
          setBusy(false);
          setError(err?.message || "Image optimization failed");
        }
        return null;
      }
    },
    [release]
  );

  const loadFile = useCallback(
    async (file) => {
      if (!file || !file.type?.startsWith("image/")) {
        setError("Please select an image file.");
        return null;
      }
      try {
        const dataUrl = await fileToDataUrl(file);
        setOriginal({ file, dataUrl, name: file.name, bytes: file.size });
        setCropped(null);
        return await run(file, fit);
      } catch (err) {
        setError(err?.message || "Could not read the image");
        return null;
      }
    },
    [fit, run]
  );

  const setFitMode = useCallback(
    async (nextFit) => {
      setFit(nextFit);
      const source = cropped?.file || original?.file;
      if (!source) return null;
      return run(source, nextFit, { croppedSource: Boolean(cropped) });
    },
    [cropped, original, run]
  );

  const setClaheEnabled = useCallback(
    async (enabled) => {
      const next = Boolean(enabled);
      claheRef.current = next;
      setClahe(next);
      const source = cropped?.file || original?.file;
      if (!source) return null;
      return run(source, fit, { croppedSource: Boolean(cropped) });
    },
    [cropped, original, fit, run]
  );

  const applyCrop = useCallback(
    async (file) => {
      if (!file) return null;
      try {
        const dataUrl = await fileToDataUrl(file);
        setCropped({ file, dataUrl });
        return await run(file, fit, { croppedSource: true });
      } catch (err) {
        setError(err?.message || "Could not apply the crop");
        return null;
      }
    },
    [fit, run]
  );

  /** Drop a manual crop and re-optimize the original upload (downscale only). */
  const clearCrop = useCallback(async () => {
    setCropped(null);
    if (!original?.file) return null;
    return run(original.file, fit, { croppedSource: false });
  }, [original, fit, run]);

  /** Adopt an already-optimized camera capture (no full-resolution original). */
  const adoptResult = useCallback(
    (file, dataUrl, meta) => {
      release();
      setOriginal({ file, dataUrl, name: file?.name || "capture.jpg", bytes: file?.size || 0 });
      setCropped(null);
      setResult({ file, dataUrl, meta });
      if (meta?.fitMode) {
        setFit({ fitMode: meta.fitMode, padStyle: meta.padStyle || "imagenet_mean" });
      }
      if (typeof meta?.clahe === "boolean") {
        claheRef.current = meta.clahe;
        setClahe(meta.clahe);
      }
      setError(null);
      setBusy(false);
    },
    [release]
  );

  const reset = useCallback(() => {
    requestRef.current += 1;
    release();
    setOriginal(null);
    setCropped(null);
    setResult(null);
    setFit(DEFAULT_FIT);
    claheRef.current = false;
    setClahe(false);
    setError(null);
    setBusy(false);
  }, [release]);

  return {
    original,
    cropped,
    result,
    fit,
    clahe,
    busy,
    error,
    loadFile,
    setFitMode,
    setClaheEnabled,
    applyCrop,
    clearCrop,
    adoptResult,
    reset,
    hasOriginal: Boolean(original),
  };
}
