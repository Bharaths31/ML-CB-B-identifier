import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { BREED_TRAITS, getBreedTraits, compareTraits, TRAIT_FIELDS } from "../../data/breedTraits.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const configPath = path.resolve(here, "../../../../src/config.py");
const configSource = fs.readFileSync(configPath, "utf8");

function pythonTuple(name) {
  const match = configSource.match(new RegExp(`${name}\\s*=\\s*\\(([\\s\\S]*?)\\)`));
  if (!match) throw new Error(`Could not find ${name}`);
  return [...match[1].matchAll(/'([^']+)'/g)].map((entry) => entry[1]);
}

const cattle = pythonTuple("EXPECTED_CATTLE_BREEDS");
const buffalo = pythonTuple("EXPECTED_BUFFALO_BREEDS");
const configured = [...cattle, ...buffalo];

test("configured class map is 50 cattle + 18 buffalo", () => {
  assert.equal(cattle.length, 50);
  assert.equal(buffalo.length, 18);
  assert.equal(configured.length, 68);
});

test("all 68 configured breeds have curated morphology", () => {
  const missing = configured.filter((key) => !getBreedTraits(key));
  assert.deepEqual(missing, [], `missing traits: ${missing.join(", ")}`);
  assert.ok(Object.keys(BREED_TRAITS).length >= 68);
});

test("every curated breed has all eight display fields", () => {
  const required = [...TRAIT_FIELDS.map((field) => field.key), "origin", "keyDistinguisher"];
  for (const key of configured) {
    const traits = getBreedTraits(key);
    for (const field of required) {
      assert.ok(traits[field], `${key} is missing ${field}`);
    }
  }
});

test("look-alike comparison covers the Rajputana trio", () => {
  const { breeds, rows } = compareTraits(["Gir", "Red Sindhi", "Sahiwal"]);
  assert.equal(breeds.length, 3);
  assert.ok(rows.find((row) => row.key === "coat").differs);
});
