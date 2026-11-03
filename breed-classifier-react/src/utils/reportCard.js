/**
 * Printable / shareable "farmer report card".
 *
 * Opens a self-contained print window (no html2canvas / PDF dependency) with
 * the animal photo, timestamp, Big-4 predictions and the care summary. The
 * farmer can print it or "Save as PDF" from the browser dialog — which is what
 * dairy cooperatives and veterinary students actually hand over.
 */

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function list(items, render) {
  if (!Array.isArray(items) || items.length === 0) return "";
  return `<ul>${items.map((item) => `<li>${render(item)}</li>`).join("")}</ul>`;
}

export function buildReportHtml({
  result,
  advisory,
  quality,
  imageUrl,
  filename,
  labels = {},
  lang = "en",
}) {
  const L = (key, fallback) => labels[key] || fallback;
  const species = result?.predicted_species || result?.species || "—";
  const breed = (result?.predicted_breed || result?.top_breed || "—").replace(/_/g, " ");
  const gender = result?.gender?.label || result?.gender?.short_label || "—";
  const age = result?.age_group?.label || "—";
  const milk = advisory?.production?.milk;
  const breedInfo = advisory?.breed || result?.knowledge?.breed || {};
  const stamp = new Date().toLocaleString();

  const rows = [
    [L("hero.species", "Species"), species],
    [L("hero.breed", "Breed"), breed],
    [L("hero.gender", "Gender"), gender],
    [L("hero.age", "Age group"), age],
    [L("hero.origin", "Native tract"), breedInfo.origin_region || "—"],
  ];

  const nutrition = advisory?.nutrition || {};
  const feeding = [
    [L("advisory.greenFodder", "Green fodder"), nutrition.green_fodder],
    [L("advisory.dryFodder", "Dry fodder"), nutrition.dry_fodder],
    [L("advisory.concentrate", "Concentrate"), nutrition.concentrate],
    [L("advisory.minerals", "Mineral mixture"), nutrition.mineral_mixture],
  ].filter(([, value]) => value);

  return `<!doctype html>
<html lang="${esc(lang)}">
<head>
<meta charset="utf-8" />
<title>${L("result.downloadCard", "Farmer report card")} — ${esc(breed)}</title>
<style>
  :root { --forest:#1B4332; --emerald:#2D6A4F; --amber:#D97706; --sand:#F8FAF5; }
  * { box-sizing: border-box; }
  body { margin:0; padding:24px; font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
         background:var(--sand); color:#14231c; }
  .card { max-width:820px; margin:0 auto; background:#fff; border-radius:18px;
          overflow:hidden; box-shadow:0 12px 40px rgba(27,67,50,.14); }
  header { background:linear-gradient(135deg,var(--forest),var(--emerald)); color:#fff;
           padding:22px 26px; }
  header h1 { margin:0 0 4px; font-size:22px; letter-spacing:.2px; }
  header p { margin:0; opacity:.82; font-size:13px; }
  .body { display:grid; grid-template-columns: 240px 1fr; gap:20px; padding:22px 26px; }
  .photo img { width:100%; border-radius:12px; display:block; border:1px solid #e5e7eb; }
  .photo small { display:block; margin-top:8px; color:#64748b; font-size:11px; line-height:1.4; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .cell { border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px; }
  .cell span { display:block; font-size:10px; text-transform:uppercase; letter-spacing:.06em;
               color:#64748b; margin-bottom:3px; }
  .cell strong { font-size:14px; }
  .section { padding:0 26px 20px; }
  .section h2 { font-size:13px; text-transform:uppercase; letter-spacing:.08em;
                color:var(--forest); margin:18px 0 8px; }
  ul { margin:0; padding-left:20px; }
  li { margin:4px 0; font-size:13px; line-height:1.5; }
  .milk { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
  .milk div { background:#ecfdf5; border:1px solid #a7f3d0; border-radius:10px; padding:10px 12px; }
  .milk span { display:block; font-size:10px; color:#065f46; text-transform:uppercase; }
  .milk strong { font-size:15px; color:#064e3b; }
  .badges { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
  .badge { font-size:11px; padding:4px 10px; border-radius:999px; background:#f1f5f9;
           border:1px solid #e2e8f0; }
  footer { padding:16px 26px 22px; border-top:1px solid #e5e7eb; color:#64748b; font-size:11px;
           line-height:1.5; }
  @media print {
    body { padding:0; background:#fff; }
    .card { box-shadow:none; border-radius:0; max-width:none; }
    @page { margin: 12mm; }
  }
</style>
</head>
<body>
  <div class="card">
    <header>
      <h1>${L("report.title", "Livestock Health & Pashu Aadhaar Card")}</h1>
      <p>${esc(stamp)} · ${esc(filename || "")}</p>
    </header>
    <div class="body">
      <div class="photo">
        ${imageUrl ? `<img src="${esc(imageUrl)}" alt="${esc(breed)}" />` : ""}
        ${quality?.overallScore != null
          ? `<small>Photo quality: ${Math.round(quality.overallScore)}/100</small>` : ""}
        <div class="badges">
          <span class="badge">${esc(species)}</span>
          <span class="badge">${esc(breed)}</span>
        </div>
      </div>
      <div class="grid">
        ${rows.map(([label, value]) => `
          <div class="cell"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("")}
      </div>
    </div>

    ${milk ? `
      <div class="section">
        <h2>${L("advisory.production", "Production")}</h2>
        <div class="milk">
          <div><span>${L("advisory.dailyYield", "Daily yield")}</span><strong>${esc(milk.daily_yield)}</strong></div>
          <div><span>${L("advisory.lactation", "Lactation")}</span><strong>${esc(milk.lactation_yield)}</strong></div>
          <div><span>${L("advisory.fat", "Fat")}</span><strong>${esc(milk.fat)}</strong></div>
        </div>
      </div>` : ""}

    ${feeding.length ? `
      <div class="section">
        <h2>${L("advisory.nutrition", "Daily feeding")}</h2>
        <div class="grid">
          ${feeding.map(([label, value]) => `
            <div class="cell"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`).join("")}
        </div>
      </div>` : ""}

    ${Array.isArray(advisory?.priority_actions) && advisory.priority_actions.length ? `
      <div class="section">
        <h2>${L("advisory.actions", "Priority actions")}</h2>
        ${list(advisory.priority_actions, esc)}
      </div>` : ""}

    ${Array.isArray(advisory?.schemes) && advisory.schemes.length ? `
      <div class="section">
        <h2>${L("advisory.schemes", "Government schemes")}</h2>
        ${list(advisory.schemes.slice(0, 5), esc)}
      </div>` : ""}

    <footer>
      ${esc(advisory?.disclaimer || L("advisory.disclaimer",
        "General guidance only — confirm with your veterinarian or State Animal Husbandry Department."))}
      <br />BreedID · ${esc(species)} · ${esc(breed)} · ${esc(gender)} · ${esc(age)}
    </footer>
  </div>
</body>
</html>`;
}

export function downloadFarmerReport(options) {
  if (typeof window === "undefined") return false;
  const html = buildReportHtml(options);
  const win = window.open("", "_blank", "width=920,height=1200");
  if (!win) return false;
  win.document.open();
  win.document.write(html);
  win.document.close();
  const trigger = () => {
    try {
      win.focus();
      win.print();
    } catch {
      /* user can print manually */
    }
  };
  if (win.document.readyState === "complete") {
    setTimeout(trigger, 250);
  } else {
    win.onload = () => setTimeout(trigger, 250);
  }
  return true;
}
