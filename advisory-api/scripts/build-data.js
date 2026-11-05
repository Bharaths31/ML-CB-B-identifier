/**
 * Build the compact advisory knowledge base from the project's curated
 * `data/breed_knowledge.json`.
 *
 * Run from this folder:
 *   node scripts/build-data.js
 *
 * The generated file is committed so the Vercel function has no build step.
 */
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.resolve(__dirname, "..", "..");
const SOURCE = path.join(ROOT, "data", "breed_knowledge.json");
const TARGET = path.resolve(__dirname, "..", "data", "breed_knowledge.json");

const CATTLE = [
  "amritmahal", "ayrshire", "bachaur", "badri", "bargur", "bhelai", "dagri",
  "dangi", "deoni", "gangatari", "gaolao", "ghumsari", "gir", "hallikar",
  "hariana", "himachali_pahari", "kangayam", "kankrej", "kenkatha", "khariar",
  "kherigarh", "khillari", "konkan_kapila", "kosali", "krishna_valley",
  "ladakhi", "lakhimi", "malnad_gidda", "malvi", "mewati", "motu", "nagori",
  "nari", "nimari", "ongole", "poda_thirupu", "ponwar", "pulikulam",
  "punganur", "purnea", "rathi", "red_kandhari", "red_sindhi", "sahiwal",
  "shweta_kapila", "siri", "tharparkar", "thutho", "umblachery", "vechur",
];

const BUFFALO = [
  "banni", "bargur", "bhadwari", "chhattisgarhi", "chilika", "gojri",
  "jaffarabadi", "kalahandi", "luit", "marathwada", "mehsana", "murrah",
  "nagpuri", "nili_ravi", "other_buffalo", "pandharpuri", "surti", "toda",
];

const ALIASES = {
  belahi: "bhelai",
  gangatiri: "gangatari",
  ghumusari: "ghumsari",
  khillar: "khillari",
  poda_thurpu: "poda_thirupu",
  bhadawari: "bhadwari",
  marathwadi: "marathwada",
  alambadi: "other_buffalo",
  mixed_buffalo: "other_buffalo",
};

function normalizeUtility(value) {
  const text = String(value || "").toLowerCase();
  if (text.includes("dual")) return "Dual";
  if (text.includes("draught") || text.includes("draft")) return "Draught";
  if (text.includes("milch") || text.includes("dairy")) return "Milch";
  return "Dual";
}

function schemeCodes(schemes) {
  const codes = new Set();
  for (const scheme of schemes || []) {
    const text = String(scheme).toLowerCase();
    if (text.includes("gokul") || /\brgm\b/.test(text)) codes.add("rgm");
    if (text.includes("national livestock mission") || /\bnlm\b/.test(text)) codes.add("nlm");
    if (text.includes("nadcp") || text.includes("fmd")) codes.add("nadcp");
    if (text.includes("kisan credit") || /\bkcc\b/.test(text)) codes.add("kcc");
    if (text.includes("dairy entrepreneurship") || text.includes("nedp")) codes.add("nedp");
    if (text.includes("insurance")) codes.add("insurance");
  }
  if (codes.size === 0) codes.add("nlm");
  return [...codes];
}

function compactEntry(entry, species) {
  const local = entry.local_names || {};
  const milk = entry.milk_yield || {};
  const feeding = entry.feeding_and_care || {};
  const climate = entry.climate_and_disease_resistance || {};
  return {
    slug: entry.slug,
    name: entry.display_name,
    species,
    local_names: {
      en: local.en || entry.display_name,
      ta: local.ta || entry.display_name,
      hi: local.hi || entry.display_name,
    },
    origin: entry.origin_region || "",
    utility_type: normalizeUtility(entry.utility_type),
    utility: normalizeUtility(entry.utility_type),
    milk_yield: milk.daily_liters
      ? {
          daily_liters: milk.daily_liters,
          daily_range: milk.daily_liters_range || null,
          lactation_yield: milk.lactation_yield || null,
          lactation_range: milk.lactation_kg_range || null,
          fat_percentage: milk.fat_percentage || null,
          fat_range: milk.fat_range || null,
        }
      : null,
    physical_markers: entry.physical_markers || [],
    feeding: {
      green_fodder_kg_per_day: feeding.green_fodder_kg_per_day || null,
      dry_fodder_kg_per_day: feeding.dry_fodder_kg_per_day || null,
      concentrate_kg_per_day: feeding.concentrate_kg_per_day || null,
      mineral_mixture_g_per_day: feeding.mineral_mixture_g_per_day || null,
      water_liters_per_day: feeding.water_liters_per_day || null,
      notes: feeding.notes || "",
    },
    climate: {
      heat_tolerance: climate.heat_tolerance || "",
      tick_parasite_resistance: climate.tick_parasite_resistance || "",
      disease_notes: climate.disease_notes || "",
    },
    market_price: entry.estimated_market_price_inr || "",
    price_range_inr: entry.price_range_inr || null,
    schemes: entry.govt_schemes || [],
    scheme_codes: schemeCodes(entry.govt_schemes || []),
    data_confidence: entry.data_confidence || "regional_default",
    note: entry.note || "",
  };
}

const source = JSON.parse(fs.readFileSync(SOURCE, "utf8"));
const breeds = { Cattle: {}, Buffalo: {} };

for (const slug of CATTLE) {
  const entry = source.Cattle?.[slug];
  if (!entry) throw new Error(`Missing cattle breed: ${slug}`);
  breeds.Cattle[slug] = compactEntry(entry, "Cattle");
}
for (const slug of BUFFALO) {
  const entry = source.Buffalo?.[slug];
  if (!entry) throw new Error(`Missing buffalo breed: ${slug}`);
  breeds.Buffalo[slug] = compactEntry(entry, "Buffalo");
}

const output = {
  meta: {
    schema_version: 2,
    breed_count: CATTLE.length + BUFFALO.length,
    cattle_count: CATTLE.length,
    buffalo_count: BUFFALO.length,
    languages: ["en", "ta", "hi"],
    generated_at: new Date().toISOString(),
    source: "data/breed_knowledge.json",
    data_policy:
      "Curated NBAGR/ICAR breed descriptions and Indian feeding norms. " +
      "Yields, prices and vaccination timings are typical ranges; confirm locally.",
  },
  aliases: ALIASES,
  breeds,
};

fs.writeFileSync(TARGET, `${JSON.stringify(output, null, 2)}\n`, "utf8");
console.log(
  `Wrote ${output.meta.breed_count} breeds (${CATTLE.length} cattle, ${BUFFALO.length} buffalo) to ${TARGET}`,
);
