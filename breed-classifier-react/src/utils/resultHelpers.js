/** Shared display helpers for compact and advanced prediction views. */

export function normalizeTop(result) {
  const raw = result?.top_k || result?.top5_breeds || [];
  const seen = new Set();
  const out = [];
  for (const item of Array.isArray(raw) ? raw : []) {
    const name = (item.breed || item.name || "").replace(/_/g, " ");
    if (!name || seen.has(name.toLowerCase())) continue;
    seen.add(name.toLowerCase());
    const prob =
      typeof item.prob === "number"
        ? item.prob
        : typeof item.confidence === "number"
        ? item.confidence / 100
        : 0;
    out.push({ name, prob });
  }
  return out;
}

export function oodTriggers(details) {
  if (!details) return [];
  const triggers = [];
  if (details.ood_by_energy) triggers.push("energy");
  if (details.ood_by_msp) triggers.push("msp");
  if (details.ood_by_binary_floor) triggers.push("floor");
  return triggers;
}

export function qualityHintKey(status) {
  switch (status) {
    case "TOO_BLURRY":
      return "tooBlurry";
    case "TOO_DARK":
      return "tooDark";
    case "OVEREXPOSED":
      return "overexposed";
    case "NO_SUBJECT_OR_TOO_FAR":
      return "noSubject";
    case "PARTIALLY_OUT_OF_FRAME":
      return "partial";
    default:
      return "ready";
  }
}
