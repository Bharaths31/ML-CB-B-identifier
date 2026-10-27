import test from "node:test";
import assert from "node:assert/strict";
import { applyClaheToImageData, TARGET_SIZE } from "../imageOptimizer.js";

function makeImage(width, height, fn) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      const [r, g, b] = fn(x, y);
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  return { data, width, height };
}

test("CLAHE preserves the 260x260 model-input dimensions", () => {
  const source = makeImage(TARGET_SIZE, TARGET_SIZE, (x, y) => {
    // Deliberately harsh left-to-right shadow gradient.
    const v = Math.round((x / TARGET_SIZE) * 120 + (y % 5));
    return [v, v, v];
  });
  const result = applyClaheToImageData(source, TARGET_SIZE, TARGET_SIZE, 2.0, 8);
  assert.equal(result.width, 260);
  assert.equal(result.height, 260);
  assert.equal(result.data.length, 260 * 260 * 4);
  assert.equal(result.data[3], 255, "alpha channel is preserved");
});

test("CLAHE lifts the dark half of a backlit frame", () => {
  const source = makeImage(TARGET_SIZE, TARGET_SIZE, (x) => {
    const v = x < TARGET_SIZE / 2 ? 18 : 210;
    return [v, v, v];
  });
  const before = source.data[0];
  const result = applyClaheToImageData(source, TARGET_SIZE, TARGET_SIZE, 2.0, 8);
  const after = result.data[0];
  assert.ok(after > before, `expected shadow lift, got ${before} -> ${after}`);
});

test("CLAHE returns a new buffer without mutating the source", () => {
  const source = makeImage(32, 32, (x, y) => [x * 8, y * 8, 64]);
  const snapshot = Uint8ClampedArray.from(source.data);
  applyClaheToImageData(source, 32, 32, 2.0, 4);
  assert.deepEqual(Array.from(source.data), Array.from(snapshot));
});
