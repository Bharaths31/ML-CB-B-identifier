"use strict";

const { findBreed, listBreeds, speciesCounts, data } = require("../lib/knowledge");
const { buildAdvisory, breedFacts, resolveLang, LANGS } = require("../lib/i18n");

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Accept",
  "Access-Control-Max-Age": "86400",
};

function normalizeGender(value) {
  const text = String(value || "").trim().toLowerCase();
  if (text.includes("female") || text.includes("பெண்") || text.includes("मादा")) return "Female";
  if (text.includes("male") || text.includes("ஆண்") || text.includes("नर")) return "Male";
  return "Unknown";
}

function normalizeAge(value) {
  const text = String(value || "").trim().toLowerCase();
  if (text.includes("calf") || text.includes("கன்று") || text.includes("बछड़")) return "Calf";
  if (text.includes("young") || text.includes("இளம்") || text.includes("युवा")) return "Young";
  if (text.includes("adult") || text.includes("முதிர்ந்த") || text.includes("वयस्क")) return "Adult";
  return "Adult";
}

function send(res, status, payload) {
  res.statusCode = status;
  for (const [key, value] of Object.entries(CORS_HEADERS)) res.setHeader(key, value);
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", status === 200
    ? "public, s-maxage=86400, stale-while-revalidate=604800"
    : "no-store");
  res.end(JSON.stringify(payload));
}

function suggestionFor(query) {
  const needle = String(query || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  if (!needle) return [];
  return listBreeds()
    .filter((item) =>
      item.slug.includes(needle.replace(/\s+/g, "_")) ||
      item.name.toLowerCase().includes(needle))
    .slice(0, 8);
}

module.exports = async function handler(req, res) {
  const started = Date.now();

  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    for (const [key, value] of Object.entries(CORS_HEADERS)) res.setHeader(key, value);
    res.end();
    return;
  }
  if (req.method !== "GET" && req.method !== "HEAD") {
    send(res, 405, {
      ok: false,
      error: "method_not_allowed",
      message: "Use GET /api/advisory?species=Cattle&breed=gir&gender=Female&age=Adult&lang=en",
    });
    return;
  }

  const query = req.query || {};
  const species = query.species || query.species_name;
  const breed = query.breed || query.breed_slug;
  const lang = resolveLang(query.lang || query.language);
  const gender = normalizeGender(query.gender || query.sex);
  const age = normalizeAge(query.age || query.age_group || query.ageGroup);

  if (!species || !breed) {
    send(res, 400, {
      ok: false,
      error: "missing_parameters",
      message: "Both 'species' and 'breed' are required.",
      example: "/api/advisory?species=Cattle&breed=gir&gender=Female&age=Adult&lang=en",
      supported_languages: LANGS,
    });
    return;
  }

  const found = findBreed(species, breed);
  if (!found) {
    send(res, 404, {
      ok: false,
      error: "unknown_breed",
      message: `No knowledge entry for species='${species}' breed='${breed}'.`,
      suggestions: suggestionFor(breed),
      counts: speciesCounts(),
    });
    return;
  }

  const facts = breedFacts(found.entry, lang);
  const advisory = buildAdvisory({
    entry: found.entry,
    gender,
    age,
    lang,
  });

  const payload = {
    ok: true,
    query: {
      species: found.species,
      breed: found.slug,
      gender,
      age,
      lang,
    },
    breed: {
      slug: found.slug,
      ...facts,
    },
    advisory,
    meta: {
      source: "static-breed-knowledge",
      generated_at: data.meta.generated_at,
      latency_ms: Date.now() - started,
      breed_count: data.meta.breed_count,
      counts: speciesCounts(),
      supported_languages: LANGS,
      disclaimer: advisory.disclaimer,
    },
  };

  if (req.method === "HEAD") {
    res.statusCode = 200;
    for (const [key, value] of Object.entries(CORS_HEADERS)) res.setHeader(key, value);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end();
    return;
  }

  send(res, 200, payload);
};
