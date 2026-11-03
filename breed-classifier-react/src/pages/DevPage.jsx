import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import DevModelPanel from "../components/DevModelPanel.jsx";
import DevOODPanel from "../components/DevOODPanel.jsx";
import DevConfigPanel from "../components/DevConfigPanel.jsx";
import DevDatasetPanel from "../components/DevDatasetPanel.jsx";
import ThemeToggle from "../components/ThemeToggle.jsx";
import LanguageToggle from "../components/LanguageToggle.jsx";
import { useI18n } from "../i18n/I18nProvider.jsx";
import { getLogs } from "../api.js";
import { useSocket } from "../hooks/useSocket.js";
import { useSharedConfig } from "../hooks/useSharedConfig.js";

const VET_PORTAL_URL =
  import.meta.env.VITE_VET_PORTAL_URL || "http://localhost:5174";

export default function DevPage() {
  const { t } = useI18n();
  const [activeTab, setActiveTab] = useState("models"); // 'models' | 'ood' | 'config' | 'logs'
  const [logs, setLogs] = useState("");
  const [logsLoading, setLogsLoading] = useState(false);

  const { connected, serverStatus, socket } = useSocket();
  const { config, updateConfig, resetConfig, refreshConfig } = useSharedConfig(socket);

  const fetchSystemLogs = async () => {
    try {
      setLogsLoading(true);
      const text = await getLogs();
      setLogs(text);
    } catch (err) {
      setLogs("Failed to fetch system logs: " + err.message);
    } finally {
      setLogsLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === "logs") {
      fetchSystemLogs();
    }
  }, [activeTab]);

  return (
    <div className="dev-page-container">
      {/* Dev Header */}
      <header className="dev-top-header">
        <div className="dev-header-left">
          <Link to="/" className="btn btn-outline back-link-btn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="19" y1="12" x2="5" y2="12"></line>
              <polyline points="12 19 5 12 12 5"></polyline>
            </svg>
            {t("dev.userAppView")}
          </Link>
          <div className="dev-title-block">
            <h1 className="dev-heading">{t("dev.title")}</h1>
            <span className="dev-subheading">{t("dev.subtitle")}</span>
          </div>
        </div>

        <div className="dev-header-right">
          <div className="status-badge" title={connected ? "WebSocket Connected" : "Connecting..."}>
            <span className={`status-dot ${connected ? "status-online" : "status-offline"}`} />
            <span className="status-text">{connected ? "Online" : "Disconnected"}</span>
          </div>

          <div className="system-spec-pills">
            <span className="spec-pill">
              {t("dev.device")}: <strong>{serverStatus?.device?.toUpperCase() || "CPU"}</strong>
            </span>
            <span className="spec-pill">
              {t("dev.models")}: <strong>{serverStatus?.models_count || 0}</strong>
            </span>
            <span className="spec-pill">
              {t("dev.uptime")}: <strong>{serverStatus?.uptime ? `${Math.round(serverStatus.uptime)}s` : "0s"}</strong>
            </span>
            <span className="spec-pill spec-pill-active" title={config?.active_model || serverStatus?.active_model || "default"}>
              {t("dev.active")}: <strong>{config?.active_model || serverStatus?.active_model || "default"}</strong>
            </span>
          </div>

          <LanguageToggle />
          <ThemeToggle />
        </div>
      </header>

      {/* Main Dev Layout */}
      <div className="dev-layout-body">
        {/* Subnavigation Bar */}
        <nav className="dev-subnav-bar">
          <button
            type="button"
            className={`dev-nav-btn ${activeTab === "models" ? "dev-nav-active" : ""}`}
            onClick={() => setActiveTab("models")}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
              <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
              <line x1="12" y1="22.08" x2="12" y2="12"></line>
            </svg>
            {t("dev.tabs.models")}
          </button>

          <button
            type="button"
            className={`dev-nav-btn ${activeTab === "datasets" ? "dev-nav-active" : ""}`}
            onClick={() => setActiveTab("datasets")}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
            </svg>
            {t("dev.tabs.dataset")}
          </button>

          <button
            type="button"
            className={`dev-nav-btn ${activeTab === "ood" ? "dev-nav-active" : ""}`}
            onClick={() => setActiveTab("ood")}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path>
              <line x1="12" y1="9" x2="12" y2="13"></line>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
            {t("dev.tabs.ood")}
          </button>

          <button
            type="button"
            className={`dev-nav-btn ${activeTab === "config" ? "dev-nav-active" : ""}`}
            onClick={() => setActiveTab("config")}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
            {t("dev.tabs.config")}
          </button>

          <button
            type="button"
            className={`dev-nav-btn ${activeTab === "logs" ? "dev-nav-active" : ""}`}
            onClick={() => setActiveTab("logs")}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="16" y1="13" x2="8" y2="13"></line>
              <line x1="16" y1="17" x2="8" y2="17"></line>
              <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
            {t("dev.tabs.logs")}
          </button>

          <Link to="/?mode=camera" className="dev-nav-btn dev-nav-link" title="Open the user capture flow">
            Camera Test
          </Link>
          <a
            href={VET_PORTAL_URL}
            target="_blank"
            rel="noreferrer"
            className="dev-nav-btn dev-nav-link"
            title="Open the standalone Veterinary Review Portal"
          >
            Vet Portal ↗
          </a>
        </nav>

        {/* Tab Panels */}
        <div className="dev-main-panel">
          {activeTab === "models" && (
            <DevModelPanel
              activeModel={config?.active_model || serverStatus?.active_model}
              onModelChanged={refreshConfig}
            />
          )}

          {activeTab === "datasets" && (
            <DevDatasetPanel />
          )}

          {activeTab === "ood" && (
            <DevOODPanel
              config={config}
              onConfigChanged={refreshConfig}
            />
          )}

          {activeTab === "config" && (
            <DevConfigPanel
              config={config}
              onUpdateConfig={updateConfig}
              onResetConfig={resetConfig}
            />
          )}

          {activeTab === "logs" && (
            <div className="dev-panel-content">
              <div className="panel-header-block">
                <div className="title-row-between">
                  <h2 className="panel-title">Inference & Training Server Logs</h2>
                  <button
                    type="button"
                    className="btn btn-sm btn-secondary"
                    onClick={fetchSystemLogs}
                    disabled={logsLoading}
                  >
                    {logsLoading ? "Refreshing..." : "Refresh Logs"}
                  </button>
                </div>
              </div>
              <div className="logs-terminal-box">
                <pre className="logs-terminal-text">
                  {logs || "No logs available."}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
