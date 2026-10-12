import { useState, useEffect, useCallback } from "react";
import { getSharedConfig, updateSharedConfig, resetSharedConfig } from "../api.js";

export function useSharedConfig(socket = null) {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getSharedConfig();
      setConfig(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  // Listen to real-time WebSocket updates
  useEffect(() => {
    const s = socket?.current || socket;
    if (!s || typeof s.on !== "function") return;

    const handleConfigChanged = (payload) => {
      if (payload?.config) {
        setConfig(payload.config);
      } else {
        fetchConfig();
      }
    };

    const handleServerStatus = (payload) => {
      if (payload?.config) {
        setConfig(payload.config);
      }
    };

    s.on("config_changed", handleConfigChanged);
    s.on("server_status", handleServerStatus);

    return () => {
      s.off("config_changed", handleConfigChanged);
      s.off("server_status", handleServerStatus);
    };
  }, [socket, fetchConfig]);

  const update = async (patch) => {
    try {
      const res = await updateSharedConfig(patch);
      if (res?.config) {
        setConfig(res.config);
      }
      return res;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const reset = async () => {
    try {
      const res = await resetSharedConfig();
      if (res?.config) {
        setConfig(res.config);
      }
      return res;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  return { config, loading, error, updateConfig: update, resetConfig: reset, refreshConfig: fetchConfig };
}
