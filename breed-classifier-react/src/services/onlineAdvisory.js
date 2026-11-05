/**
 * Online Breed Tips / farmer advisory client.
 *
 * Matches the Vercel serverless function in `advisory-api/api/advisory.js`.
 * The function never throws: the Android UI always gets a plain result object.
 */

export const ADVISORY_TIMEOUT_MS = 5000;

const ENV_ADVISORY_API_URL = String(
  (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_ADVISORY_API_URL) || "",
).trim();

const OFFLINE_MESSAGES = {
  en: "You are offline. Showing on-device breed identification only. Connect to the internet for breed tips.",
  ta: "நீங்கள் ஆஃப்லைனில் உள்ளீர்கள். சாதனத்தில் இனம் கண்டறிதல் மட்டும் கிடைக்கும். இனக் குறிப்புகளுக்கு இணையத்துடன் இணைக்கவும்.",
  hi: "आप ऑफ़लाइन हैं। केवल डिवाइस पर नस्ल पहचान उपलब्ध है। नस्ल सुझावों के लिए इंटरनेट से जुड़ें।",
};

const TIMEOUT_MESSAGES = {
  en: "Breed tips took too long to load. The on-device result is still valid; try again when the network is stable.",
  ta: "இனக் குறிப்புகள் ஏற்ற அதிக நேரமாயிற்று. சாதனத்தில் கிடைத்த முடிவு செல்லுபடியாகும்; நெட்வொர்க் சீரானதும் மீண்டும் முயற்சிக்கவும்.",
  hi: "नस्ल सुझाव लोड होने में बहुत समय लगा। डिवाइस पर मिला परिणाम मान्य है; नेटवर्क स्थिर होने पर फिर कोशिश करें।",
};

const ERROR_MESSAGES = {
  en: "Breed tips are unavailable right now. The on-device result is still valid.",
  ta: "இனக் குறிப்புகள் இப்போது கிடைக்கவில்லை. சாதனத்தில் கிடைத்த முடிவு செல்லுபடியாகும்.",
  hi: "नस्ल सुझाव अभी उपलब्ध नहीं हैं। डिवाइस पर मिला परिणाम मान्य है।",
};

function normalizeLang(lang) {
  const value = String(lang || "en").toLowerCase().slice(0, 2);
  return value === "ta" || value === "hi" ? value : "en";
}

function messageFor(table, lang) {
  return table[normalizeLang(lang)] || table.en;
}

function normalizeBase(apiUrl) {
  const raw = String(apiUrl || ENV_ADVISORY_API_URL || "").trim();
  if (!raw) return "";
  return /^https?:\/\//i.test(raw) ? raw : `https://${raw}`;
}

function buildUrl(apiUrl, species, breedSlug, gender, ageGroup, lang) {
  const base = normalizeBase(apiUrl);
  if (!base) return null;
  let url;
  try {
    url = new URL(base);
  } catch {
    return null;
  }
  if (!/\/api\/advisory\/?$/i.test(url.pathname)) {
    url.pathname = `${url.pathname.replace(/\/+$/, "")}/api/advisory`;
  }
  url.searchParams.set("species", String(species || "Cattle"));
  url.searchParams.set("breed", String(breedSlug || ""));
  url.searchParams.set("gender", String(gender || "Unknown"));
  url.searchParams.set("age", String(ageGroup || "Adult"));
  url.searchParams.set("lang", normalizeLang(lang));
  return url.toString();
}

/**
 * Fetch breed tips from the free Vercel advisory API.
 *
 * @param {string} species "Cattle" | "Buffalo"
 * @param {string} breedSlug e.g. "gir", "murrah"
 * @param {string} gender "Female" | "Male" | "Unknown"
 * @param {string} ageGroup "Calf" | "Young" | "Adult"
 * @param {string} lang "en" | "ta" | "hi"
 * @param {string} apiUrl Full endpoint or project base URL
 * @returns {Promise<object>} never throws
 */
export async function fetchAdvisoryData(species, breedSlug, gender, ageGroup, lang, apiUrl) {
  const language = normalizeLang(lang);

  if (typeof navigator !== "undefined" && navigator.onLine === false) {
    return {
      ok: false,
      offline: true,
      configured: true,
      message: messageFor(OFFLINE_MESSAGES, language),
    };
  }

  const url = buildUrl(apiUrl, species, breedSlug, gender, ageGroup, language);
  if (!url) {
    return {
      ok: false,
      offline: false,
      configured: false,
      message: messageFor(ERROR_MESSAGES, language),
      error: "missing_advisory_api_url",
    };
  }

  const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = controller
    ? setTimeout(() => controller.abort(), ADVISORY_TIMEOUT_MS)
    : null;

  try {
    const response = await fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
      signal: controller ? controller.signal : undefined,
    });

    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }

    if (!response.ok) {
      return {
        ok: false,
        offline: false,
        configured: true,
        status: response.status,
        error: payload?.error || "advisory_http_error",
        message: payload?.message || messageFor(ERROR_MESSAGES, language),
      };
    }

    return {
      ok: true,
      offline: false,
      configured: true,
      ...(payload || {}),
    };
  } catch (error) {
    const timedOut = error?.name === "AbortError";
    return {
      ok: false,
      offline: false,
      configured: true,
      timeout: timedOut,
      error: timedOut ? "advisory_timeout" : "advisory_network_error",
      message: messageFor(timedOut ? TIMEOUT_MESSAGES : ERROR_MESSAGES, language),
    };
  } finally {
    if (timer) clearTimeout(timer);
  }
}

export { ENV_ADVISORY_API_URL };
