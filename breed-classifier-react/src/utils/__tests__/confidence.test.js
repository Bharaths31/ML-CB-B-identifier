import test from "node:test";
import assert from "node:assert/strict";
import {
  CONFIDENCE_LEVELS,
  confidenceMeta,
  formatBreedConfidence,
  getConfidenceLevel,
  resolveConfidenceLevel,
} from "../confidence.js";

test("five-level rubric maps the joint confidence thresholds", () => {
  assert.equal(getConfidenceLevel(0.99), 5);
  assert.equal(getConfidenceLevel(0.85), 5);
  assert.equal(getConfidenceLevel(0.849), 4);
  assert.equal(getConfidenceLevel(0.75), 4);
  assert.equal(getConfidenceLevel(0.74), 3);
  assert.equal(getConfidenceLevel(0.6), 3);
  assert.equal(getConfidenceLevel(0.59), 2);
  assert.equal(getConfidenceLevel(0.4), 2);
  assert.equal(getConfidenceLevel(0.399), 1);
  assert.equal(getConfidenceLevel(0.99, true), 1, "OOD is always level 1");
});

test("rubric metadata exposes badge colours for all five levels", () => {
  for (const level of [1, 2, 3, 4, 5]) {
    const meta = CONFIDENCE_LEVELS[level];
    assert.equal(meta.level, level);
    assert.ok(meta.badge.startsWith("#"));
    assert.ok(meta.actionKey.startsWith("confidence.level"));
  }
  assert.equal(confidenceMeta(5).colour, "dark_green");
  assert.equal(confidenceMeta(2).colour, "orange");
});

test("server-reported level wins, absent levels are recomputed", () => {
  assert.equal(resolveConfidenceLevel({ confidence: 0.9, confidence_level: 4 }), 4);
  assert.equal(resolveConfidenceLevel({ confidence: 0.3, confidence_level: 3, is_ood: true }), 1);
  assert.equal(resolveConfidenceLevel({ confidence: 0.8 }), 4);
});

test("presenter confidence modes mirror the standalone tester", () => {
  const scaled = formatBreedConfidence(40, [40, 30, 20], "scaled", 80);
  assert.equal(scaled.value, "80.0%");
  const clamped = formatBreedConfidence(40, [40], "clamped", 80);
  assert.equal(clamped.value, "80.0%");
  const hidden = formatBreedConfidence(40, [40], "hide", 80);
  assert.equal(hidden.hidden, true);
  const badge = formatBreedConfidence(40, [40], "badge_only", 80);
  assert.equal(badge.value, "Primary Match");
  const raw = formatBreedConfidence(40, [40], "percentage", 80);
  assert.equal(raw.value, "40.0%");
});
