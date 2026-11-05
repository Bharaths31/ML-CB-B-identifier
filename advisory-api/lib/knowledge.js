"use strict";

const data = require("../data/breed_knowledge.json");

const SPECIES_ALIASES = {
  cattle: "Cattle",
  cow: "Cattle",
  கால்நடை: "Cattle",
  गाय: "Cattle",
  buffalo: "Buffalo",
  bhains: "Buffalo",
  எருமை: "Buffalo",
  भैंस: "Buffalo",
};

function normalizeSpecies(value) {
  const text = String(value || "").trim().toLowerCase();
  if (SPECIES_ALIASES[text]) return SPECIES_ALIASES[text];
  if (text.includes("buffalo")) return "Buffalo";
  if (text.includes("cattle") || text.includes("cow")) return "Cattle";
  return null;
}

function normalizeBreed(value) {
  const raw = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return data.aliases[raw] || raw;
}

function findBreed(speciesValue, breedValue) {
  const species = normalizeSpecies(speciesValue);
  const slug = normalizeBreed(breedValue);
  if (!species || !slug) return null;
  const entry = data.breeds[species]?.[slug];
  if (!entry) return null;
  return { species, slug, entry };
}

function listBreeds() {
  const rows = [];
  for (const species of Object.keys(data.breeds)) {
    for (const [slug, entry] of Object.entries(data.breeds[species])) {
      rows.push({
        species,
        slug,
        name: entry.name,
        local_names: entry.local_names,
      });
    }
  }
  return rows;
}

function speciesCounts() {
  return Object.fromEntries(
    Object.entries(data.breeds).map(([species, entries]) => [species, Object.keys(entries).length]),
  );
}

module.exports = {
  data,
  normalizeSpecies,
  normalizeBreed,
  findBreed,
  listBreeds,
  speciesCounts,
};
