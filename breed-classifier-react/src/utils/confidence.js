/**
 * Five-level confidence rubric + presenter confidence formatting.
 *
 * C = P(species) × P(breed | species), i.e. the soft-routed top-1
 * probability already returned as `confidence` in 0..1.
 */

export const CONFIDENCE_LEVELS = {
  1: {
    level: 1,
    key: "confidence.level1.label",
    actionKey: "confidence.level1.action",
    band: "very_low",
    colour: "red",
    badge: "#ef4444",
    acceptRegistration: false,
    escalate: true,
  },
  2: {
    level: 2,
    key: "confidence.level2.label",
    actionKey: "confidence.level2.action",
    band: "low",
    colour: "orange",
    badge: "#f97316",
    acceptRegistration: false,
    escalate: true,
  },
  3: {
    level: 3,
    key: "confidence.level3.label",
    actionKey: "confidence.level3.action",
    band: "moderate",
    colour: "amber",
    badge: "#f59e0b",
    acceptRegistration: true,
    escalate: false,
  },
  4: {
    level: 4,
    key: "confidence.level4.label",
    actionKey: "confidence.level4.action",
    band: "high",
    colour: "light_green",
    badge: "#84cc16",
    acceptRegistration: true,
    escalate: false,
  },
  5: {
    level: 5,
    key: "confidence.level5.label",
    actionKey: "confidence.level5.action",
    band: "very_high",
    colour: "dark_green",
    badge: "#15803d",
    acceptRegistration: true,
    escalate: false,
  },
};

export function clampConfidence(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.min(1, Math.max(0, n));
}

/** Map C (0..1) to 1..5. OOD is always Level 1. */
export function getConfidenceLevel(confidence, isOod = false) {
  if (isOod) return 1;
  const c = clampConfidence(confidence);
  if (c >= 0.85) return 5;
  if (c >= 0.75) return 4;
  if (c >= 0.6) return 3;
  if (c >= 0.4) return 2;
  return 1;
}

/** Prefer the server's authoritative level, but recompute when absent. */
export function resolveConfidenceLevel(result) {
  const isOod = Boolean(result?.is_ood || result?.ood);
  const reported = Number(result?.confidence_level);
  if (Number.isInteger(reported) && reported >= 1 && reported <= 5) {
    return isOod ? 1 : reported;
  }
  return getConfidenceLevel(result?.confidence, isOod);
}

export function confidenceMeta(levelOrResult) {
  const level = typeof levelOrResult === "object"
    ? resolveConfidenceLevel(levelOrResult)
    : Number(levelOrResult) || 1;
  return CONFIDENCE_LEVELS[level] || CONFIDENCE_LEVELS[1];
}

/**
 * Port of the standalone tester's presenter formatting.
 * @param {number} rawPercent 0..100 raw top-1 confidence
 * @param {number[]} top3Percents raw percentages of the top 3 candidates
 * @param {string} mode percentage|scaled|clamped|badge_only|hide
 * @param {number} floorPct floor used by scaled/clamped modes
 */
export function formatBreedConfidence(rawPercent, top3Percents = [], mode = "percentage", floorPct = 80) {
  const raw = Number.isFinite(Number(rawPercent)) ? Number(rawPercent) : 0;
  const floor = Math.min(99.5, Math.max(0, Number(floorPct) || 0));
  switch (mode) {
    case "hide":
      return { value: "Identified", labelKey: "result.confidenceStatus", hidden: true };
    case "badge_only":
      return {
        value: raw >= 25 ? "Primary Match" : "Probable Match",
        labelKey: "result.confidenceMatch",
        hidden: true,
      };
    case "clamped":
      return {
        value: `${Math.max(floor, raw).toFixed(1)}%`,
        labelKey: "result.confidence",
        hidden: false,
      };
    case "scaled": {
      const sum = (top3Percents || []).slice(0, 3).reduce((s, v) => s + (Number(v) || 0), 0);
      let scaled = sum > 0 ? (raw / sum) * 100 : raw;
      scaled = Math.min(99.4, Math.max(floor, scaled));
      return { value: `${scaled.toFixed(1)}%`, labelKey: "result.confidence", hidden: false };
    }
    case "percentage":
    default:
      return { value: `${raw.toFixed(1)}%`, labelKey: "result.confidence", hidden: false };
  }
}
