import { useState, useEffect, useRef } from "react";

export const THEMES = [
  {
    id: "dark",
    name: "Midnight Dark",
    icon: "🌙",
    bgPreview: "#0b0f19",
    accentPreview: "#6366f1",
  },
  {
    id: "light",
    name: "Clean Light",
    icon: "☀️",
    bgPreview: "#f8fafc",
    accentPreview: "#4f46e5",
  },
  {
    id: "emerald",
    name: "Pasture Green",
    icon: "🌿",
    bgPreview: "#061a14",
    accentPreview: "#10b981",
  },
  {
    id: "sunset",
    name: "Warm Sunset",
    icon: "🌅",
    bgPreview: "#19110b",
    accentPreview: "#f59e0b",
  },
  {
    id: "cyber",
    name: "Cyber Neon",
    icon: "🌌",
    bgPreview: "#0d081f",
    accentPreview: "#06b6d4",
  },
];

const THEME_KEY = "breed-classifier-theme";

export default function ThemeToggle() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem(THEME_KEY) || "dark";
  });
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    if (open) {
      document.addEventListener("mousedown", handleOutside);
    }
    return () => document.removeEventListener("mousedown", handleOutside);
  }, [open]);

  const currentTheme = THEMES.find((t) => t.id === theme) || THEMES[0];

  return (
    <div className="theme-toggle-wrapper" ref={dropdownRef}>
      <button
        className="theme-toggle-btn"
        onClick={() => setOpen(!open)}
        title="Change UI Theme"
        aria-label="Change theme"
        aria-expanded={open}
      >
        <span className="theme-btn-icon">{currentTheme.icon}</span>
        <span className="theme-btn-label">{currentTheme.name}</span>
        <span className="theme-arrow">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="theme-dropdown-menu">
          <div className="theme-dropdown-header">Select Theme</div>
          <div className="theme-options-list">
            {THEMES.map((t) => {
              const active = t.id === theme;
              return (
                <button
                  key={t.id}
                  className={`theme-option-item ${active ? "active" : ""}`}
                  onClick={() => {
                    setTheme(t.id);
                    setOpen(false);
                  }}
                >
                  <div className="theme-preview-dots">
                    <span
                      className="preview-dot bg"
                      style={{ background: t.bgPreview }}
                    />
                    <span
                      className="preview-dot accent"
                      style={{ background: t.accentPreview }}
                    />
                  </div>
                  <span className="theme-option-icon">{t.icon}</span>
                  <span className="theme-option-name">{t.name}</span>
                  {active && <span className="theme-check">✓</span>}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
