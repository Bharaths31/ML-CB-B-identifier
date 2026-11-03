/**
 * UI localization + theme integrity audit (run by tests/run_vv_suite.py).
 *
 * Usage: node src/verify_ui_i18n.mjs
 * Output: single-line JSON {ok, checks[], metrics, problems[]}
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { LANGUAGES, TRANSLATIONS } from "./i18n/translations.js";
import {
  BREED_NAMES_I18N,
  TRAIT_FIELDS_I18N,
  TRAIT_VALUES_I18N,
  DYNAMIC_I18N,
} from "./i18n/dictionaries.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const PROJECT = path.resolve(ROOT, "..");

const checks = [];
const problems = [];

function check(id, name, ok, detail = "", metric = null) {
  checks.push({
    id,
    name,
    status: ok ? "PASS" : "FAIL",
    detail,
    metric,
  });
  if (!ok) problems.push(`${id}: ${detail || name}`);
}

const TAMIL = /[\u0B80-\u0BFF]/;
const DEVANAGARI = /[\u0900-\u097F]/;
const LATIN = /[A-Za-z]/;

const en = TRANSLATIONS.en || {};
const ta = TRANSLATIONS.ta || {};
const hi = TRANSLATIONS.hi || {};

/* 1 — key parity ---------------------------------------------------------- */
const enKeys = Object.keys(en);
const missingTa = enKeys.filter((k) => !(k in ta) || !String(ta[k] ?? "").trim());
const missingHi = enKeys.filter((k) => !(k in hi) || !String(hi[k] ?? "").trim());

check("i18n.parity.ta", "Tamil key parity (en → ta)", missingTa.length === 0,
  missingTa.length ? `missing ${missingTa.length}: ${missingTa.slice(0, 5).join(", ")}` : `${enKeys.length}/${enKeys.length} keys`,
  { total: enKeys.length, missing: missingTa.length });
check("i18n.parity.hi", "Hindi key parity (en → hi)", missingHi.length === 0,
  missingHi.length ? `missing ${missingHi.length}: ${missingHi.slice(0, 5).join(", ")}` : `${enKeys.length}/${enKeys.length} keys`,
  { total: enKeys.length, missing: missingHi.length });

check("i18n.languages", "Language switcher exposes en / ta / hi",
  LANGUAGES.length === 3 && LANGUAGES.every((l) => l.id && l.label),
  LANGUAGES.map((l) => `${l.id}:${l.label}`).join(" | "));

/* 2 — script validation --------------------------------------------------- */
const CRITICAL_PREFIXES = [
  "hero.", "advisory.", "photoGuide.", "photoCheck.", "traitCheck.",
  "encyclopedia.", "result.tabs.", "quality.badge.", "attributes.",
];

function scriptAudit(bucket, regex, label, emptyName) {
  const latinOnly = [];
  const criticalLatin = [];
  for (const [key, raw] of Object.entries(bucket)) {
    const value = String(raw ?? "");
    if (!value.trim()) continue;
    const hasScript = regex.test(value);
    const templateOnly = /\{\w+\}/.test(value)
      && !/[A-Za-z]/.test(value.replace(/\{\w+\}/g, ""));
    if (!hasScript && LATIN.test(value) && !templateOnly) {
      latinOnly.push(key);
      if (CRITICAL_PREFIXES.some((p) => key.startsWith(p))) criticalLatin.push(key);
    }
  }
  const ratio = 1 - latinOnly.length / Math.max(1, Object.keys(bucket).length);
  check(`i18n.script.${label}`, `${emptyName} script coverage`, criticalLatin.length === 0,
    criticalLatin.length
      ? `farmer-critical keys still Latin: ${criticalLatin.slice(0, 6).join(", ")}`
      : `${(ratio * 100).toFixed(1)}% of values contain ${emptyName} script`,
    { ratio: Number(ratio.toFixed(4)), latinOnly: latinOnly.length, critical: criticalLatin.length });
  return criticalLatin;
}

scriptAudit(ta, TAMIL, "ta", "Tamil");
scriptAudit(hi, DEVANAGARI, "hi", "Devanagari");

/* 3 — Big-4 labels & traits ---------------------------------------------- */
const HERO_KEYS = [
  "hero.species", "hero.breed", "hero.gender", "hero.age",
  "advisory.female", "advisory.male",
  "advisory.calf", "advisory.young", "advisory.adult",
];
const heroMissing = [];
for (const key of HERO_KEYS) {
  const tv = String(ta[key] ?? "");
  const hv = String(hi[key] ?? "");
  if (!TAMIL.test(tv) || !DEVANAGARI.test(hv)) heroMissing.push(key);
}
check("i18n.hero", "Big-4 species/gender/age labels in ta + hi",
  heroMissing.length === 0,
  heroMissing.length ? `untranslated: ${heroMissing.join(", ")}` : `${HERO_KEYS.length} labels localized`,
  { labels: HERO_KEYS.length, missing: heroMissing.length });

const MODEL_TRAIT_FIELDS = ["hump", "horn", "coat", "ear", "dewlap", "face", "size"];
const traitFieldFailures = MODEL_TRAIT_FIELDS
  .filter((field) => {
    const entry = TRAIT_FIELDS_I18N[field];
    return !entry || !TAMIL.test(entry.ta || "") || !DEVANAGARI.test(entry.hi || "");
  });
check("i18n.traits.fields", "All 7 model trait labels localized",
  traitFieldFailures.length === 0,
  traitFieldFailures.length
    ? `missing: ${traitFieldFailures.join(", ")}`
    : `${MODEL_TRAIT_FIELDS.length} model traits + ${Object.keys(TRAIT_FIELDS_I18N).length - MODEL_TRAIT_FIELDS.length} display-only field(s)`,
  { fields: Object.keys(TRAIT_FIELDS_I18N).length, model_fields: MODEL_TRAIT_FIELDS.length });

const untranslatedTraitValues = Object.entries(TRAIT_VALUES_I18N)
  .filter(([, v]) => !TAMIL.test(v.ta || "") || !DEVANAGARI.test(v.hi || "")).length;
check("i18n.traits.values", "Trait value dictionary localized",
  untranslatedTraitValues === 0,
  `${Object.keys(TRAIT_VALUES_I18N).length - untranslatedTraitValues}/${Object.keys(TRAIT_VALUES_I18N).length} values have ta + hi`,
  { total: Object.keys(TRAIT_VALUES_I18N).length, untranslated: untranslatedTraitValues });

/* 4 — 68-breed dictionary ------------------------------------------------- */
let knowledge = { Cattle: {}, Buffalo: {} };
try {
  knowledge = JSON.parse(
    fs.readFileSync(path.join(PROJECT, "data", "breed_knowledge.json"), "utf8")
  );
} catch (err) {
  problems.push(`breed knowledge unreadable: ${err.message}`);
}

const breedSlugs = [
  ...Object.keys(knowledge.Cattle || {}),
  ...Object.keys(knowledge.Buffalo || {}),
];
const breedMissing = breedSlugs.filter((slug) => {
  const entry = BREED_NAMES_I18N[slug];
  return !entry || !TAMIL.test(entry.ta || "") || !DEVANAGARI.test(entry.hi || "");
});

check("i18n.breeds", "70 breed names localized (68 + crossbred)",
  breedSlugs.length >= 68 && breedMissing.length === 0,
  breedMissing.length
    ? `missing ${breedMissing.length}: ${breedMissing.slice(0, 6).join(", ")}`
    : `${breedSlugs.length}/${breedSlugs.length} breeds mapped`,
  { breeds: breedSlugs.length, missing: breedMissing.length });

/* 5 — dynamic value maps -------------------------------------------------- */
const dynamicFails = [];
for (const kind of ["species", "gender", "age"]) {
  for (const [value, entry] of Object.entries(DYNAMIC_I18N[kind] || {})) {
    if (!TAMIL.test(entry.ta || "") || !DEVANAGARI.test(entry.hi || "")) {
      dynamicFails.push(`${kind}.${value}`);
    }
  }
}
check("i18n.dynamic", "Species / gender / age dynamic values localized",
  dynamicFails.length === 0,
  dynamicFails.length ? `untranslated: ${dynamicFails.slice(0, 6).join(", ")}` : "all dynamic labels localized",
  { missing: dynamicFails.length });

/* 6 — theme tokens -------------------------------------------------------- */
const themeCss = fs.readFileSync(path.join(HERE, "styles", "themes.css"), "utf8");
const THEMES = ["field-parchment", "obsidian-loam", "clinical-slate", "night-telemetry"];
const REQUIRED_VARS = [
  "--bg-canvas", "--bg-surface", "--bg-elevated", "--bg-recessed",
  "--text-primary", "--text-secondary", "--text-muted",
  "--border-subtle", "--border-focus",
  "--accent-primary", "--accent-secondary", "--accent-surface", "--status-ok",
];
const themeProblems = [];
for (const id of THEMES) {
  const idx = themeCss.indexOf(`[data-theme="${id}"]`);
  if (idx === -1) {
    themeProblems.push(`${id}: selector missing`);
    continue;
  }
  const nextIdx = themeCss.indexOf("[data-theme=", idx + 1);
  const block = themeCss.slice(idx, nextIdx === -1 ? undefined : nextIdx);
  for (const variable of REQUIRED_VARS) {
    if (!block.includes(variable)) themeProblems.push(`${id}: ${variable}`);
  }
}
check("themes.tokens", "4 themes define the full token set",
  themeProblems.length === 0 && themeCss.includes("Noto Sans Tamil")
    && themeCss.includes("Noto Sans Devanagari")
    && themeCss.includes("Plus Jakarta Sans")
    && themeCss.includes("JetBrains Mono"),
  themeProblems.length ? themeProblems.slice(0, 6).join(" ; ") : "field-parchment / obsidian-loam / clinical-slate / night-telemetry",
  { themes: THEMES.length, variables: REQUIRED_VARS.length });

/* ------------------------------------------------------------------------ */
const total = checks.length;
const failed = checks.filter((c) => c.status === "FAIL").length;

console.log(JSON.stringify({
  ok: failed === 0,
  checks,
  problems,
  metrics: {
    languages: LANGUAGES.length,
    enKeys: enKeys.length,
    taKeys: Object.keys(ta).length,
    hiKeys: Object.keys(hi).length,
    breeds: breedSlugs.length,
    themes: THEMES.length,
  },
}));

process.exit(failed === 0 ? 0 : 1);
