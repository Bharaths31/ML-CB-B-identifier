import { useEffect, useState } from "react";
import { resolveApiBase, rememberBackendUrl } from "../services/apiBase.js";

export function useBackendHealth({ enabled, backendUrl, online }) {
  const [status, setStatus] = useState("offline");

  useEffect(() => {
    if (!enabled || !online) {
      setStatus("offline");
      return undefined;
    }

    const base = resolveApiBase(backendUrl);
    if (!base) {
      setStatus("unconfigured");
      return undefined;
    }

    let cancelled = false;
    let timer = null;

    const check = async () => {
      setStatus((current) => (current === "connected" ? "connected" : "connecting"));
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      try {
        const response = await fetch(`${base}/api/health`, {
          headers: { Accept: "application/json" },
          signal: controller.signal,
        });
        if (!cancelled) setStatus(response.ok ? "connected" : "reconnecting");
        if (response.ok) rememberBackendUrl(base);
      } catch {
        if (!cancelled) setStatus("reconnecting");
      } finally {
        clearTimeout(timeout);
      }
    };

    check();
    timer = setInterval(check, 15000);

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [enabled, backendUrl, online]);

  return status;
}
