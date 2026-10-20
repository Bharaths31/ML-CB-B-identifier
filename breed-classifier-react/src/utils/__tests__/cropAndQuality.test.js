import test from "node:test";
import assert from "node:assert/strict";
import {
  QUALITY_THRESHOLDS,
  analyzeImageData,
  centerSquareRect,
  clampPan,
  computeCoverScale,
  computeSourceRect,
} from "../cropAndQuality.js";

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

test("cover scale matches CSS object-fit: cover", () => {
  assert.equal(computeCoverScale(1920, 1080, 400, 300), 300 / 1080);
  assert.equal(computeCoverScale(1080, 1920, 400, 300), 400 / 1080);
});

test("landscape 1920x1080 -> centred 1:1 crop", () => {
  const rect = computeSourceRect({
    viewportW: 400,
    viewportH: 300,
    sourceW: 1920,
    sourceH: 1080,
    reticleX: 83,
    reticleY: 33,
    reticleSize: 234,
  });
  assert.ok(rect.size > 800 && rect.size <= 1080);
  assert.ok(rect.x >= 0 && rect.x + rect.size <= 1920);
  assert.ok(rect.y >= 0 && rect.y + rect.size <= 1080);
  // The crop must be horizontally centred in the source.
  assert.ok(Math.abs(rect.x + rect.size / 2 - 960) < 1.5);
});

test("portrait 1080x1920 -> centred 1:1 crop", () => {
  const rect = computeSourceRect({
    viewportW: 400,
    viewportH: 300,
    sourceW: 1080,
    sourceH: 1920,
    reticleX: 83,
    reticleY: 33,
    reticleSize: 234,
  });
  assert.ok(rect.size > 500 && rect.size <= 1080);
  assert.ok(Math.abs(rect.y + rect.size / 2 - 960) < 1.5);
});

test("zoom shrinks the crop around the same centre", () => {
  const base = {
    viewportW: 400,
    viewportH: 300,
    sourceW: 1920,
    sourceH: 1080,
    reticleX: 83,
    reticleY: 33,
    reticleSize: 234,
  };
  const z1 = computeSourceRect({ ...base, zoom: 1 });
  const z2 = computeSourceRect({ ...base, zoom: 2 });
  assert.ok(Math.abs(z2.size - z1.size / 2) < 0.01);
  assert.ok(Math.abs(z2.x + z2.size / 2 - (z1.x + z1.size / 2)) < 0.01);
  assert.ok(Math.abs(z2.y + z2.size / 2 - (z1.y + z1.size / 2)) < 0.01);
});

test("pan moves the crop and stays clamped", () => {
  const base = {
    viewportW: 400,
    viewportH: 300,
    sourceW: 1920,
    sourceH: 1080,
    reticleX: 83,
    reticleY: 33,
    reticleSize: 234,
    zoom: 1,
  };
  const centred = computeSourceRect(base);
  const pannedRight = computeSourceRect({ ...base, panX: 80 });
  assert.ok(pannedRight.x < centred.x, "dragging the image right moves the crop left");

  const extreme = computeSourceRect({ ...base, panX: 99999, panY: 99999 });
  assert.equal(extreme.x, 0);
  assert.equal(extreme.y, 0);
});

test("front-camera mirror inverts the x coordinate", () => {
  const base = {
    viewportW: 400,
    viewportH: 300,
    sourceW: 1920,
    sourceH: 1080,
    reticleX: 83,
    reticleY: 33,
    reticleSize: 234,
  };
  const normal = computeSourceRect(base);
  const mirrored = computeSourceRect({ ...base, mirrored: true });
  assert.ok(Math.abs(mirrored.x - (1920 - normal.x - normal.size)) < 0.001);
  assert.ok(Math.abs(mirrored.size - normal.size) < 0.001);
});

test("crop never exceeds the shortest source side", () => {
  const rect = computeSourceRect({
    viewportW: 300,
    viewportH: 300,
    sourceW: 320,
    sourceH: 240,
    reticleX: 0,
    reticleY: 0,
    reticleSize: 300,
    zoom: 1,
  });
  assert.ok(rect.size <= 240 + 1e-9);
  assert.ok(rect.x >= 0 && rect.x + rect.size <= 320 + 1e-9);
  assert.ok(rect.y >= 0 && rect.y + rect.size <= 240 + 1e-9);
});

test("clampPan bounds pan to the covered image", () => {
  const clamped = clampPan({
    panX: 9999,
    panY: -9999,
    zoom: 2,
    viewportW: 400,
    viewportH: 300,
    sourceW: 1920,
    sourceH: 1080,
  });
  assert.ok(clamped.x <= (1920 * (300 / 1080) * 2 - 400) / 2 + 1e-6);
  assert.ok(clamped.y >= -(1080 * (300 / 1080) * 2 - 300) / 2 - 1e-6);
});

test("centerSquareRect matches shortest-side + centre crop", () => {
  assert.deepEqual(centerSquareRect(1920, 1080), { x: 420, y: 0, size: 1080 });
});

// ---------------------------------------------------------------------------
// Quality gate
// ---------------------------------------------------------------------------

function buffer(width, height, pixel) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = pixel(x, y);
      const i = (y * width + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  return { data, width, height };
}

test("flat grey frame is blurry and has no subject", () => {
  const report = analyzeImageData(buffer(130, 130, () => [128, 128, 128]));
  assert.equal(report.sharpnessScore, 0);
  assert.ok(report.flags.includes("TOO_BLURRY"));
  assert.ok(report.flags.includes("NO_SUBJECT_OR_TOO_FAR"));
  assert.equal(report.isReady, false);
});

test("high-contrast checkerboard passes sharpness and framing", () => {
  const report = analyzeImageData(
    buffer(130, 130, (x, y) => ((Math.floor(x / 6) + Math.floor(y / 6)) % 2 ? [255, 255, 255] : [0, 0, 0]))
  );
  assert.ok(report.sharpnessScore > QUALITY_THRESHOLDS.blurScoreMin, `sharpness=${report.sharpnessScore}`);
  assert.ok(report.centerEdge > QUALITY_THRESHOLDS.minCenterEdge, `centerEdge=${report.centerEdge}`);
  assert.ok(report.framingScore > 0);
});

test("all-black and all-white frames are flagged for exposure", () => {
  const dark = analyzeImageData(buffer(130, 130, () => [0, 0, 0]));
  const bright = analyzeImageData(buffer(130, 130, () => [255, 255, 255]));
  assert.ok(dark.flags.includes("TOO_DARK"));
  assert.ok(bright.flags.includes("OVEREXPOSED"));
  assert.ok(dark.exposureScore < 50);
  assert.ok(bright.exposureScore < 50);
});

test("quality scores stay in 0..100", () => {
  const report = analyzeImageData(
    buffer(130, 130, (x, y) => [x % 256, (x * 3) % 256, (y * 5) % 256])
  );
  for (const key of ["overallScore", "sharpnessScore", "exposureScore", "framingScore"]) {
    assert.ok(report[key] >= 0 && report[key] <= 100, `${key}=${report[key]}`);
  }
});
