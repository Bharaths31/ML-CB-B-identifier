export const ENV_BACKEND_URL = String(
  import.meta.env.VITE_BACKEND_URL || ""
).trim().replace(/\/+$/, "");

const LAST_BACKEND_KEY = "breedid_last_backend_url";

function normalize(url) {
  return String(url || "").trim().replace(/\/+$/, "");
}

export function getLastBackendUrl() {
  try {
    const value = normalize(localStorage.getItem(LAST_BACKEND_KEY) || "");
    return /^https?:\/\//i.test(value) ? value : "";
  } catch {
    return "";
  }
}

export function rememberBackendUrl(url) {
  const value = normalize(url);
  if (!/^https?:\/\//i.test(value)) return;
  try {
    localStorage.setItem(LAST_BACKEND_KEY, value);
  } catch {
    /* storage can be blocked */
  }
}

export function resolveApiBase(configured = "") {
  const explicit = normalize(configured);
  if (explicit) return explicit;
  if (ENV_BACKEND_URL) return ENV_BACKEND_URL;

  const remembered = getLastBackendUrl();
  if (remembered) return remembered;

  if (typeof window === "undefined") return "";

  const { protocol, hostname, port, origin } = window.location;
  if (!/^https?:$/.test(protocol)) return "";
  if (hostname === "localhost" && !port) return "";
  if (hostname === "127.0.0.1" && !port) return "";

  return origin.replace(/\/+$/, "");
}
