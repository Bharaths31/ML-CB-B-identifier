import test from "node:test";
import assert from "node:assert/strict";
import { BREED_DATASET_COUNTS, DATASET_SUMMARY } from "../../data/breedDatasetCounts.js";

test("dataset inventory exists and has summary metadata", () => {
  assert.ok(Array.isArray(BREED_DATASET_COUNTS), "BREED_DATASET_COUNTS must be an array");
  assert.ok(BREED_DATASET_COUNTS.length > 50, "should catalog more than 50 breeds");
  assert.ok(DATASET_SUMMARY.totalImages > 10000, "total images should exceed 10,000");
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

test("both cattle and buffalo are represented in the dataset inventory", () => {
  const cattle = BREED_DATASET_COUNTS.filter((b) => b.species === "cattle");
  const buffalo = BREED_DATASET_COUNTS.filter((b) => b.species === "buffalo");

  assert.ok(cattle.length >= 50, "must include at least 50 cattle breeds");
  assert.ok(buffalo.length >= 18, "must include at least 18 buffalo breeds");
});

test("every breed entry has required schema fields", () => {
  for (const b of BREED_DATASET_COUNTS) {
    assert.ok(typeof b.key === "string" && b.key.length > 0, "key must be a non-empty string");
    assert.ok(typeof b.name === "string" && b.name.length > 0, "name must be a non-empty string");
    assert.ok(b.species === "cattle" || b.species === "buffalo", "species must be cattle or buffalo");
    assert.ok(typeof b.count === "number" && b.count >= 0, "count must be non-negative integer");
    assert.ok(typeof b.tier === "string" && b.tier.length > 0, "tier must be present");
  }
});

test("Gir is the top represented breed with highest count", () => {
  assert.equal(BREED_DATASET_COUNTS[0].key, "gir");
  assert.equal(BREED_DATASET_COUNTS[0].name, "Gir");
  assert.equal(BREED_DATASET_COUNTS[0].count, 1121);
});

