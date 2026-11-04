import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  LANGUAGES,
  TRANSLATIONS,
  translate,
} from "../../i18n/translations.js";
import {
  BREED_NAMES_I18N,
  TRAIT_FIELDS_I18N,
  TRAIT_VALUES_I18N,
  DYNAMIC_I18N,
  localizeBreed,
  localizeDynamic,
  localizeTraitValue,
  localizeAdvisory,
} from "../../i18n/dictionaries.js";
import { BREED_DATASET_COUNTS } from "../../data/breedDatasetCounts.js";
import { BREED_TRAITS, TRAIT_FIELDS } from "../../data/breedTraits.js";

describe("i18n Localization & Translation Integrity", () => {
  it("supports English, Tamil and Hindi with short identifiers", () => {
    assert.equal(LANGUAGES.length, 3);
    const ids = LANGUAGES.map((l) => l.id);
    assert.deepEqual(ids, ["en", "ta", "hi"]);
    const ta = LANGUAGES.find((l) => l.id === "ta");
    assert.equal(ta.label, "தமிழ்");
  });

  it("every English translation key exists in Tamil and Hindi", () => {
    const enKeys = Object.keys(TRANSLATIONS.en);
    const taKeys = new Set(Object.keys(TRANSLATIONS.ta));
    const hiKeys = new Set(Object.keys(TRANSLATIONS.hi));

    const missingInTa = enKeys.filter((k) => !taKeys.has(k));
    const missingInHi = enKeys.filter((k) => !hiKeys.has(k));

    assert.deepEqual(missingInTa, [], "All EN keys must exist in TA");
    assert.deepEqual(missingInHi, [], "All EN keys must exist in HI");
  });

  it("translate() interpolates variables correctly in all languages", () => {
    const en = translate("en", "session.count", { count: 5 });
    assert.equal(en, "5 animals");

    const ta = translate("ta", "session.count", { count: 5 });
    assert.equal(ta, "5 விலங்குகள்");

    const hi = translate("hi", "session.count", { count: 5 });
    assert.equal(hi, "5 पशु");
  });

  it("translates Breed ID properly into native language terms rather than English or transliteration", () => {
    assert.equal(TRANSLATIONS.en["app.title"], "Breed ID • Livestock Vision");
    assert.equal(TRANSLATIONS.ta["app.title"], "இன அடையாளம் • கால்நடை பார்வை");
    assert.equal(TRANSLATIONS.hi["app.title"], "नस्ल पहचान • पशुधन दृष्टि");
    assert.ok(!TRANSLATIONS.ta["app.title"].includes("BreedID"), "Tamil app.title must not contain raw English 'BreedID'");
    assert.ok(!TRANSLATIONS.hi["app.title"].includes("ब्रीडआईडी"), "Hindi app.title must be translated, not transliterated");

    assert.ok(TRANSLATIONS.en["advanced.tabOptimizer"]);
    assert.ok(TRANSLATIONS.ta["advanced.tabOptimizer"]);
    assert.ok(TRANSLATIONS.hi["advanced.tabOptimizer"]);
    assert.ok(TRANSLATIONS.en["advanced.tabDiagnostics"]);
    assert.ok(TRANSLATIONS.ta["advanced.tabDiagnostics"]);
    assert.ok(TRANSLATIONS.hi["advanced.tabDiagnostics"]);
    assert.ok(TRANSLATIONS.en["advanced.tabBatch"]);
    assert.ok(TRANSLATIONS.ta["advanced.tabBatch"]);
    assert.ok(TRANSLATIONS.hi["advanced.tabBatch"]);

    // Camera options in advanced mode
    assert.ok(TRANSLATIONS.en["advanced.tabCamera"]);
    assert.equal(TRANSLATIONS.ta["advanced.tabCamera"], "கேமரா அமைப்புகள்");
    assert.equal(TRANSLATIONS.hi["advanced.tabCamera"], "कैमरा विकल्प");
    assert.ok(TRANSLATIONS.en["advanced.openCamera"]);
    assert.ok(TRANSLATIONS.ta["advanced.openCamera"]);
    assert.ok(TRANSLATIONS.hi["advanced.openCamera"]);
    assert.ok(TRANSLATIONS.en["advanced.uploadImage"]);
    assert.ok(TRANSLATIONS.ta["advanced.uploadImage"]);
    assert.ok(TRANSLATIONS.hi["advanced.uploadImage"]);
  });

  it("replaces transliterated loan words with authentic native Tamil livestock vocabulary", () => {
    // Check developer settings, portal, and pad translations
    assert.equal(TRANSLATIONS.ta["header.expertPortal"], "கால்நடை மருத்துவர் தளம்");
    assert.equal(TRANSLATIONS.ta["header.dev"], "தொழில்நுட்ப அமைப்புகள்");
    assert.equal(TRANSLATIONS.ta["camera.title"], "நேரடிப் படப்பிடிப்பு");
    assert.equal(TRANSLATIONS.ta["optimizer.padMean"], "நிரப்பு · ImageNet சராசரி");
    assert.equal(TRANSLATIONS.ta["optimizer.padding"], "விளிம்பு நிரப்பல்");
    assert.equal(TRANSLATIONS.ta["hero.breed"], "இனம்");
    assert.equal(TRANSLATIONS.ta["theme.label"], "வண்ணத் தோற்றம்");
    assert.equal(TRANSLATIONS.ta["encyclopedia.ticks"], "உண்ணிகள் / ஒட்டுண்ணிகள்");
    assert.equal(TRANSLATIONS.ta["result.predictedBreed"], "கணிக்கப்பட்ட இனம்");
  });

  it("localizes all 68 Atharvadarpude dataset breeds in Tamil and Hindi", () => {
    for (const item of BREED_DATASET_COUNTS) {
      const ta = localizeBreed(item.key, "ta");
      const hi = localizeBreed(item.key, "hi");
      assert.ok(ta && /[^\x00-\x7F]/.test(ta), `Breed ${item.key} must have native Tamil name, got: ${ta}`);
      assert.ok(hi && /[^\x00-\x7F]/.test(hi), `Breed ${item.key} must have native Hindi name, got: ${hi}`);
    }
  });

  it("localizes dynamic values case-insensitively for species, gender, age, utility", () => {
    // Case insensitivity
    assert.equal(localizeDynamic("species", "cattle", "ta"), "மாடு (பசு/காளை)");
    assert.equal(localizeDynamic("species", "Cattle", "ta"), "மாடு (பசு/காளை)");
    assert.equal(localizeDynamic("species", "buffalo", "ta"), "எருமை");
    assert.equal(localizeDynamic("species", "Buffalo", "ta"), "எருமை");

    // Gender
    assert.equal(localizeDynamic("gender", "female", "ta"), "பெண் (பசு)");
    assert.equal(localizeDynamic("gender", "male", "ta"), "ஆண் (காளை / எருது)");

    // Age
    assert.equal(localizeDynamic("age", "calf", "ta"), "கன்று (< 1 வருடம்)");
    assert.equal(localizeDynamic("age", "adult", "ta"), "வளர்ந்த மாடு (> 3 வருடம்)");

    // Utility
    assert.equal(localizeDynamic("utility", "Milch (Dairy)", "ta"), "கறவை மாடு (பால் பண்ணை)");
  });

  it("localizes all curated morphological traits in Tamil without falling back to raw English", () => {
    assert.equal(TRAIT_FIELDS_I18N.hump.ta, "திமில்");
    assert.equal(TRAIT_FIELDS_I18N.horn.ta, "கொம்பு");
    assert.equal(TRAIT_FIELDS_I18N.coat.ta, "உடல் நிறம்");
    assert.equal(TRAIT_FIELDS_I18N.ear.ta, "காது");
    assert.equal(TRAIT_FIELDS_I18N.dewlap.ta, "தாடைத் தோல் (அலைதாடி)");
    assert.equal(TRAIT_FIELDS_I18N.origin.ta, "தாயகம் / பூர்வீகம்");

    for (const [key, b] of Object.entries(BREED_TRAITS)) {
      for (const field of TRAIT_FIELDS) {
        if (field.key !== "origin" && b[field.key]) {
          const localized = localizeTraitValue(field.key, b[field.key], "ta");
          assert.ok(
            !/[A-Za-z]/.test(localized) || localized === b[field.key],
            `Trait ${field.key} "${b[field.key]}" should be localized in Tamil, got: ${localized}`
          );
        }
      }
    }
  });

  it("localizes advisory terms including traditional terms (சீம்பால், மடிநோய், விதை நீக்கம்)", () => {
    const mockAdvisory = {
      breed: { slug: "gir", display_name: "Gir", utility_type: "Milch (Dairy)" },
      context: { gender: "female", age_group: "adult" },
      production: { milk: { daily_yield: "12-18", fat: "4.5%" } },
      health: {
        vaccination: ["FMD"],
        watch_for: ["Mastitis"],
      },
    };

    const localizedTa = localizeAdvisory(mockAdvisory, "ta", {
      tBreed: (b) => localizeBreed(b, "ta"),
      tDynamic: (k, v) => localizeDynamic(k, v, "ta"),
    });

    assert.ok(localizedTa.headline.includes("வளர்ந்த பசு"));
    assert.ok(localizedTa.priority_actions.some((a) => a.includes("மடிநோய் (மஸ்டிடிஸ்)")));
    assert.ok(localizedTa.priority_actions.some((a) => a.includes("கறவையை நிறுத்தவும்")));

    const calfAdvisory = {
      breed: { slug: "gir", display_name: "Gir" },
      context: { gender: "female", age_group: "calf" },
    };
    const localizedCalf = localizeAdvisory(calfAdvisory, "ta", {
      tBreed: (b) => localizeBreed(b, "ta"),
      tDynamic: (k, v) => localizeDynamic(k, v, "ta"),
    });

    assert.ok(localizedCalf.priority_actions.some((a) => a.includes("சீம்பால்")));
  });

  it("includes all BPA modal and Developer control plane keys", () => {
    const bpaKeys = [
      "bpa.title",
      "bpa.subtitle",
      "bpa.predictedBreed",
      "bpa.top3",
      "bpa.earTagLabel",
      "bpa.regenerate",
      "bpa.typeCustom",
      "bpa.ownerName",
      "bpa.villageTaluk",
      "bpa.ageYears",
      "bpa.ageMonths",
      "bpa.lactationStatus",
      "bpa.flwId",
      "bpa.saveToHerd",
      "bpa.downloadOdt",
      "bpa.downloadCard",
    ];

    for (const key of bpaKeys) {
      assert.ok(TRANSLATIONS.en[key], `EN must have ${key}`);
      assert.ok(TRANSLATIONS.ta[key], `TA must have ${key}`);
      assert.ok(TRANSLATIONS.hi[key], `HI must have ${key}`);
    }

    const devKeys = [
      "dev.userAppView",
      "dev.title",
      "dev.subtitle",
      "dev.device",
      "dev.models",
      "dev.uptime",
      "dev.active",
      "dev.cameraTest",
      "dev.tabs.models",
      "dev.tabs.dataset",
      "dev.tabs.ood",
      "dev.tabs.config",
      "dev.tabs.logs",
      "dev.dataset.kpiTotal",
      "dev.dataset.kpiBreeds",
      "dev.dataset.kpiTop",
      "dev.dataset.searchPlaceholder",
      "dev.dataset.allSpecies",
      "dev.dataset.allTiers",
      "dev.dataset.exportCsv",
    ];

    for (const key of devKeys) {
      assert.ok(TRANSLATIONS.en[key], `EN must have ${key}`);
      assert.ok(TRANSLATIONS.ta[key], `TA must have ${key}`);
      assert.ok(TRANSLATIONS.hi[key], `HI must have ${key}`);
    }
  });
});

