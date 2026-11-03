import test from "node:test";
import assert from "node:assert/strict";
import { BREED_DATASET_COUNTS, DATASET_SUMMARY } from "../../data/breedDatasetCounts.js";

test("atharvadarpude dataset inventory exists and has summary metadata", () => {
  assert.ok(Array.isArray(BREED_DATASET_COUNTS), "BREED_DATASET_COUNTS must be an array");
  assert.equal(BREED_DATASET_COUNTS.length, 68, "should catalog exactly 68 breeds (50 cattle + 18 buffalo)");
  assert.equal(DATASET_SUMMARY.totalImages, 11199, "total images must match 11,199 exactly");
  assert.equal(DATASET_SUMMARY.bySpecies.cattle.images, 8531, "cattle images must equal 8,531");
  assert.equal(DATASET_SUMMARY.bySpecies.buffalo.images, 2668, "buffalo images must equal 2,668");
});

test("breeds are arranged in strictly descending order by dataset image count", () => {
  for (let i = 0; i < BREED_DATASET_COUNTS.length - 1; i++) {
    const current = BREED_DATASET_COUNTS[i].count;
    const next = BREED_DATASET_COUNTS[i + 1].count;
    assert.ok(
      current >= next,
      `Breed at index ${i} (${BREED_DATASET_COUNTS[i].name}: ${current}) must have count >= index ${i + 1} (${BREED_DATASET_COUNTS[i + 1].name}: ${next})`
    );
  }
});

test("both cattle and buffalo match the canonical 50/18 counts", () => {
  const cattle = BREED_DATASET_COUNTS.filter((b) => b.species === "cattle");
  const buffalo = BREED_DATASET_COUNTS.filter((b) => b.species === "buffalo");

  assert.equal(cattle.length, 50, "must include exactly 50 cattle breeds");
  assert.equal(buffalo.length, 18, "must include exactly 18 buffalo breeds");
});

test("every breed entry has required schema fields and valid dataset source", () => {
  for (const b of BREED_DATASET_COUNTS) {
    assert.ok(typeof b.key === "string" && b.key.length > 0, "key must be a non-empty string");
    assert.ok(typeof b.name === "string" && b.name.length > 0, "name must be a non-empty string");
    assert.ok(b.species === "cattle" || b.species === "buffalo", "species must be cattle or buffalo");
    assert.ok(typeof b.count === "number" && b.count > 0, "count must be positive integer");
    assert.ok(typeof b.tier === "string" && b.tier.length > 0, "tier must be present");
    assert.ok(b.dataset_source.startsWith("atharvadarpude/"), "dataset source must be atharvadarpude");
  }
});

test("Gir is the top represented breed with highest count", () => {
  assert.equal(BREED_DATASET_COUNTS[0].key, "gir");
  assert.equal(BREED_DATASET_COUNTS[0].name, "Gir");
  assert.equal(BREED_DATASET_COUNTS[0].count, 686);
});
