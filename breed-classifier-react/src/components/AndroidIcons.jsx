const base = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": "true",
};

export function BrandMark({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <path d="M6.5 8.5 3.5 7v4l3 1" />
      <path d="M17.5 8.5 20.5 7v4l-3 1" />
      <path d="M7 8.5h10v3.8a5 5 0 0 1-10 0V8.5Z" />
      <path d="M9.4 12h.01M14.6 12h.01" />
      <path d="M10 15h4" />
      <path d="M12 17.5V19" />
      <circle cx="12" cy="20.5" r="1.4" />
    </svg>
  );
}

export function GalleryIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <rect x="3" y="4.5" width="18" height="15" rx="2" />
      <circle cx="8.5" cy="10" r="1.4" />
      <path d="m4.5 17 4.8-4.8 3 3 2.7-2.7 4.5 4.5" />
    </svg>
  );
}

export function CameraIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <path d="M4 7h3l1.5-2h7L17 7h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z" />
      <circle cx="12" cy="13" r="3.2" />
    </svg>
  );
}

export function TrashIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <path d="M4 7h16M9 7V5h6v2M7 7l1 13h8l1-13" />
      <path d="M10 11v5M14 11v5" />
    </svg>
  );
}

export function LockIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <rect x="5" y="10" width="14" height="10" rx="2" />
      <path d="M8 10V8a4 4 0 0 1 8 0v2" />
    </svg>
  );
}

export function GlobeIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M3.5 12h17" />
      <path d="M12 3.5c2 2.3 3 5.3 3 8.5s-1 6.2-3 8.5c-2-2.3-3-5.3-3-8.5s1-6.2 3-8.5Z" />
    </svg>
  );
}

export function ThemeIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 3.5v17a8.5 8.5 0 0 0 0-17Z" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function SlidersIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <path d="M4 7h8M18 7h2M4 17h2M10 17h10" />
      <circle cx="15" cy="7" r="2.2" />
      <circle cx="7" cy="17" r="2.2" />
    </svg>
  );
}

export function LanguageIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <path d="M4 5h9M8.5 5v14M4 12h9" />
      <path d="M13.5 9h6.5l-3.2 10L13.5 9Z" />
    </svg>
  );
}

export function BookIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21.5V5.5Z" />
      <path d="M4 18.5A2.5 2.5 0 0 1 6.5 16H20" />
    </svg>
  );
}

export function ShareIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <circle cx="6" cy="12" r="2.4" />
      <circle cx="18" cy="6" r="2.4" />
      <circle cx="18" cy="18" r="2.4" />
      <path d="m8.2 10.9 7.6-3.7M8.2 13.1l7.6 3.7" />
    </svg>
  );
}

export function TargetIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <circle cx="12" cy="12" r="7" />
      <circle cx="12" cy="12" r="2" />
      <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
    </svg>
  );
}

export function FemaleIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <circle cx="12" cy="9" r="5" />
      <path d="M12 14v6M9 18h6" />
    </svg>
  );
}

export function MaleIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <circle cx="14" cy="10" r="5" />
      <path d="m15.5 5.5 4-4M16 1.5h4v4" />
    </svg>
  );
}

export function AgeIcon({ className = "" }) {
  return (
    <svg {...base} className={className}>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 7v5l3 2" />
    </svg>
  );
}
