import test from "node:test";
import assert from "node:assert/strict";
import {
  makeSessionEntry,
  sessionStats,
  sessionToCsv,
  sessionToJson,
  sessionToOdtResults,
} from "../sessionStore.js";
import { breedKey, compareTraits, getBreedTraits } from "../../data/breedTraits.js";

const result = {
  predicted_species: "Cattle",
  species: "Cattle",
  species_confidence: 93.2,
  predicted_breed: "Gir",
  top_breed: "Gir",
  confidence: 0.81,
  is_ood: false,
  ood_score: null,
  model_name: "lite2_mock",
  top5_breeds: [
    { breed: "Gir", confidence: 81 },
    { breed: "Red Sindhi", confidence: 9 },
  ],
};

test("session entry captures the export fields", () => {
  const entry = makeSessionEntry({
    result,
    quality: { overallScore: 88, status: "READY" },
    thumbnail: "data:image/jpeg;base64,xxx",
    filename: "gir.jpg",
  });
  assert.equal(entry.breed, "Gir");
  assert.equal(entry.species, "Cattle");
  assert.equal(entry.confidence, 0.81);
  assert.equal(entry.qualityScore, 88);
  assert.equal(entry.top5.length, 2);
});

test("session stats split species and average confidence", () => {
  const a = makeSessionEntry({ result, filename: "a.jpg" });
  const b = makeSessionEntry({
    result: { ...result, predicted_species: "Buffalo", species: "Buffalo", confidence: 0.5, is_ood: true },
    filename: "b.jpg",
  });
  const stats = sessionStats([a, b]);
  assert.equal(stats.total, 2);
  assert.equal(stats.cattle, 1);
  assert.equal(stats.buffalo, 1);
  assert.equal(stats.flagged, 1);
  // The OOD entry is excluded from the confidence average.
  assert.ok(Math.abs(stats.avgConfidence - 0.81) < 1e-9);
});

test("CSV escapes commas and quotes", () => {
  const entry = makeSessionEntry({
    result: { ...result, predicted_breed: 'Gir, "elite"' },
    filename: "a,b.jpg",
  });
  const csv = sessionToCsv([entry]);
  assert.ok(csv.includes('"Gir, ""elite"""'));
  assert.ok(csv.includes('"a,b.jpg"'));
});

test("JSON export includes the count", () => {
  const entry = makeSessionEntry({ result, filename: "a.jpg" });
  const parsed = JSON.parse(sessionToJson([entry]));
  assert.equal(parsed.count, 1);
  assert.equal(parsed.animals[0].breed, "Gir");
});

test("ODT mapping matches the server result schema", () => {
  const entry = makeSessionEntry({ result, filename: "a.jpg" });
  const [odt] = sessionToOdtResults([entry]);
  assert.equal(odt.top_breed, "Gir");
  assert.equal(odt.top_breed_confidence, 81);
  assert.equal(odt.species, "Cattle");
  assert.equal(odt.top5_breeds.length, 2);
});

test("breed keys normalise display names and aliases", () => {
  assert.equal(breedKey("Red Sindhi"), "red_sindhi");
  assert.equal(breedKey("Malnad Gidda"), "malnad_gidda");
  assert.equal(breedKey("Bhadwari"), "bhadawari");
  assert.equal(getBreedTraits("Gir").origin, "Gujarat, India");
});

test("trait comparison flags the rows that separate candidates", () => {
  const { breeds, rows } = compareTraits(["Gir", "Red Sindhi", "Sahiwal"]);
  assert.equal(breeds.length, 3);
  const coat = rows.find((r) => r.key === "coat");
  assert.equal(coat.differs, true);
  const origin = rows.find((r) => r.key === "origin");
  assert.equal(origin.differs, true);
});
