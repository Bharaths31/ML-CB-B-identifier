/**
 * Canonical breed catalog (must match src/config.py).
 *
 * Keep this list in sync with `EXPECTED_CATTLE_BREEDS` and
 * `EXPECTED_BUFFALO_BREEDS`. Display localization is handled by `tBreed()` from
 * the i18n provider, so only the canonical slugs live here.
 */

export const IMAGE_SIZE = 260;
export const TRAIN_RESIZE = 288;
export const IMAGENET_MEAN = [0.485, 0.456, 0.406];
export const IMAGENET_STD = [0.229, 0.224, 0.225];

export const DEFAULT_OOD = {
  enabled: true,
  energyTemperature: 1.0,
  energyThreshold: -3.2004,
  mspThreshold: 0.5863,
  binaryConfidenceFloor: 0.60,
  activeOodProfile: "auto",
};

export const EXPECTED_CATTLE_BREEDS = [
  "amritmahal", "ayrshire", "bachaur", "badri", "bargur", "bhelai", "dagri",
  "dangi", "deoni", "gangatari", "gaolao", "ghumsari", "gir", "hallikar",
  "hariana", "himachali_pahari", "kangayam", "kankrej", "kenkatha", "khariar",
  "kherigarh", "khillari", "konkan_kapila", "kosali", "krishna_valley",
  "ladakhi", "lakhimi", "malnad_gidda", "malvi", "mewati", "motu", "nagori",
  "nari", "nimari", "ongole", "poda_thirupu", "ponwar", "pulikulam",
  "punganur", "purnea", "rathi", "red_kandhari", "red_sindhi", "sahiwal",
  "shweta_kapila", "siri", "tharparkar", "thutho", "umblachery", "vechur",
];

export const EXPECTED_BUFFALO_BREEDS = [
  "banni", "bargur", "bhadwari", "chhattisgarhi", "chilika", "gojri",
  "jaffarabadi", "kalahandi", "luit", "marathwada", "mehsana", "murrah",
  "nagpuri", "nili_ravi", "other_buffalo", "pandharpuri", "surti", "toda",
];

// Dairy breeds where the visual prior leans female; used only as a soft
// heuristic by the on-device gender estimator.
export const DAIRY_BREEDS = new Set([
  "gir", "sahiwal", "red_sindhi", "tharparkar", "kankrej", "rathi",
  "mehsana", "murrah", "surti", "nili_ravi", "jaffarabadi", "banni",
]);

export function breedDisplayName(slug) {
  return String(slug || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function classifyBreedIndex(index) {
  const idx = Number(index);
  if (!Number.isFinite(idx)) return null;
  if (idx >= 0 && idx < EXPECTED_CATTLE_BREEDS.length) {
    return {
      species: "Cattle",
      slug: EXPECTED_CATTLE_BREEDS[idx],
      display: breedDisplayName(EXPECTED_CATTLE_BREEDS[idx]),
      index: idx,
    };
  }
  const buffaloIdx = idx - EXPECTED_CATTLE_BREEDS.length;
  if (buffaloIdx >= 0 && buffaloIdx < EXPECTED_BUFFALO_BREEDS.length) {
    return {
      species: "Buffalo",
      slug: EXPECTED_BUFFALO_BREEDS[buffaloIdx],
      display: breedDisplayName(EXPECTED_BUFFALO_BREEDS[buffaloIdx]),
      index: idx,
    };
  }
  return null;
}

export const GENDER_OPTIONS = [
  { key: "female", label: "Female" },
  { key: "male", label: "Male" },
];

export const AGE_OPTIONS = [
  { key: "calf", label: "Calf (< 1 yr)" },
  { key: "young", label: "Young (1-3 yrs)" },
  { key: "adult", label: "Adult (> 3 yrs)" },
];
