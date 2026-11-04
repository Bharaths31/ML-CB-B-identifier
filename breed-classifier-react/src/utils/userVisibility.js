/**
 * User-page content visibility.
 *
 * The vet portal writes `display.hidden_user_sections` (array of ids) into the
 * shared config. The user page reads it through `isUserSectionVisible()` so a
 * deployment can hide whole blocks without forking the UI.
 */

export const USER_SECTION_OPTIONS = [
  { id: "photoGuide", label: "Photo guide strip", help: "Camera tips strip above the upload card and its modal." },
  { id: "specStrip", label: "Model specification strip", help: "Target / OOD / input-size summary under the upload card." },
  { id: "herdSession", label: "Herd session", help: "Telemetry counter, session drawer and CSV/JSON export." },
  { id: "advanced", label: "Advanced panel", help: "Gear button, optimizer, X-Ray, batch tester and diagnostics." },
  { id: "devLink", label: "Developer link", help: "Link to /dev in the user header and mobile sheet." },
  { id: "escalation", label: "Escalation badge", help: "Offline escalation outbox badge and manual escalation." },
  { id: "footer", label: "Footer", help: "Status bar with model, device and version." },
  { id: "voice", label: "Voice reader", help: "Text-to-speech button in the result workspace." },
  { id: "reportCard", label: "Report card download", help: "Printable Pashu Aadhaar download button." },
  { id: "traits", label: "Trait checklist", help: "Explainable-AI morphological trait chips." },
  { id: "advisory", label: "Farmer / public tabs", help: "Advisory and breed encyclopedia tabs in the result view." },
  { id: "photoGuidance", label: "Photo guidance", help: "Photo-quality tips shown above the Big-4 result grid." },
];

export const DISPLAY_TOGGLE_OPTIONS = [
  {
    key: "show_species_badge",
    label: "Show species badge",
    help: "Display the CATTLE / BUFFALO pill in the result workspace header.",
    default: true,
  },
  {
    key: "show_species_confidence",
    label: "Show species confidence",
    help: "Display the species confidence row and meter on the Big-4 species card.",
    default: true,
  },
  {
    key: "show_escalation_warning",
    label: "Show vet escalation warning",
    help: "Show Level 1–2 vet prompts and the manual escalate action.",
    default: true,
  },
  {
    key: "show_technical_details",
    label: "Always show technical details",
    help: "Expand latency/tensor diagnostics automatically in the Advanced panel.",
    default: false,
  },
];

export function hiddenUserSections(config) {
  const value = config?.display?.hidden_user_sections;
  const configured = Array.isArray(value) ? value : [];
  // Per-link method: /?hide=herdSession,footer (useful for kiosk QR links).
  let fromUrl = [];
  try {
    if (typeof window !== "undefined") {
      const raw = new URLSearchParams(window.location.search).get("hide") || "";
      fromUrl = raw.split(",").map((item) => item.trim()).filter(Boolean);
    }
  } catch {
    fromUrl = [];
  }
  return Array.from(new Set([...configured, ...fromUrl]));
}

export function isUserSectionVisible(config, sectionId) {
  return !hiddenUserSections(config).includes(sectionId);
}

export function userVisibilitySnapshot(config) {
  const hidden = new Set(hiddenUserSections(config));
  return Object.fromEntries(USER_SECTION_OPTIONS.map((option) => [
    option.id, !hidden.has(option.id),
  ]));
}
