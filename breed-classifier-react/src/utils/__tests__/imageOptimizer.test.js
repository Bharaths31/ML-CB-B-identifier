import test from "node:test";
import assert from "node:assert/strict";
import {
  IMAGENET_MEAN_RGB,
  TARGET_SIZE,
  aspectRatio,
  computeOptimizerLayout,
  formatBytes,
} from "../imageOptimizer.js";

test("260x260 target and ImageNet mean padding colour", () => {
  assert.equal(TARGET_SIZE, 260);
  assert.deepEqual(IMAGENET_MEAN_RGB, [124, 116, 104]);
});

test("1920x1080 letterboxes to 260x146 with padY=57", () => {
  const layout = computeOptimizerLayout(1920, 1080);
  assert.equal(layout.fitMode, "pad");
  assert.equal(layout.targetWidth, 260);
  assert.equal(layout.targetHeight, 260);
  assert.equal(layout.innerWidth, 260);
  assert.equal(layout.innerHeight, 146);
  assert.equal(layout.padX, 0);
  assert.equal(layout.padY, 57);
  assert.equal(layout.scaleFactor, 260 / 1920);
  assert.equal(layout.upscaled, false);
});

test("1080x1920 letterboxes to 146x260 with padX=57", () => {
  const layout = computeOptimizerLayout(1080, 1920);
  assert.equal(layout.innerWidth, 146);
  assert.equal(layout.innerHeight, 260);
  assert.equal(layout.padX, 57);
  assert.equal(layout.padY, 0);
});

test("a 260x260 input is an identity layout", () => {
  const layout = computeOptimizerLayout(260, 260);
  assert.equal(layout.innerWidth, 260);
  assert.equal(layout.innerHeight, 260);
  assert.equal(layout.padX, 0);
  assert.equal(layout.padY, 0);
  assert.equal(layout.scaleFactor, 1);
});

test("small images are padded, not upscaled, by default", () => {
  const layout = computeOptimizerLayout(200, 100);
  assert.equal(layout.innerWidth, 200);
  assert.equal(layout.innerHeight, 100);
  assert.equal(layout.padX, 30);
  assert.equal(layout.padY, 80);
  assert.equal(layout.upscaled, false);
});

test("allowUpscale opts into filling the long side", () => {
  const layout = computeOptimizerLayout(200, 100, { allowUpscale: true });
  assert.equal(layout.innerWidth, 260);
  assert.equal(layout.innerHeight, 130);
  assert.equal(layout.padX, 0);
  assert.equal(layout.padY, 65);
  assert.equal(layout.upscaled, true);
});

test("stretch fills 260x260 with no padding", () => {
  const layout = computeOptimizerLayout(1920, 1080, { fitMode: "stretch" });
  assert.equal(layout.fitMode, "stretch");
  assert.equal(layout.innerWidth, 260);
  assert.equal(layout.innerHeight, 260);
  assert.equal(layout.padX, 0);
  assert.equal(layout.padY, 0);
  assert.ok(Math.abs(layout.scaleX - 260 / 1920) < 1e-12);
  assert.ok(Math.abs(layout.scaleY - 260 / 1080) < 1e-12);
});

test("aspect ratio reduction", () => {
  assert.equal(aspectRatio(1920, 1080), "16:9");
  assert.equal(aspectRatio(1080, 1920), "9:16");
  assert.equal(aspectRatio(260, 260), "1:1");
  assert.equal(aspectRatio(640, 480), "4:3");
  assert.equal(aspectRatio(1280, 853), "3:2");
  assert.equal(aspectRatio(1280, 961), "4:3");
});

test("byte formatting", () => {
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(2048), "2.0 KB");
  assert.equal(formatBytes(3 * 1024 * 1024), "3.00 MB");
  assert.equal(formatBytes(null), "—");
});
