import { Capacitor } from "@capacitor/core";

/** True inside the Capacitor Android/iOS WebView. */
export function isNativePlatform() {
  try {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      if (params.get("native") === "1") return true;
    }
    return Boolean(Capacitor?.isNativePlatform?.());
  } catch {
    return false;
  }
}

export const IS_NATIVE = isNativePlatform();
