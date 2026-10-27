import { useEffect, useMemo, useRef, useState } from "react";
import { getBpaQrMatrix, exportOdt, downloadBlob } from "../api.js";
import { loadImage, shareOrDownloadCanvas, roundRect } from "../utils/cropAndQuality.js";
import { makeSessionEntry, addSessionEntry, thumbnailFromDataUrl } from "../utils/sessionStore.js";

export const BPA_TAG_KEY = "breed-bpa-tag-v1";

/** 12-digit numeric NDLM/BPA-style UID (Tamil Nadu state prefix 33). */
export function generateBpaTag() {
  let digits = "33";
  for (let i = 0; i < 10; i += 1) {
    digits += Math.floor(Math.random() * 10);
  }
  return digits;
}

const SEXES = ["Female", "Male"];
const LACTATION = ["Not applicable", "Lactating", "Dry", "Pregnant", "Heifer"];
const VACCINATION = ["FMD vaccinated", "HS/BQ vaccinated", "Both", "Unknown"];

export default function BpaRegistrationModal({
  open,
  onClose,
  result,
  config,
  quality,
  location,
  modelInputUrl,
}) {
  const [tag, setTag] = useState(() => generateBpaTag());
  const [customTag, setCustomTag] = useState(false);
  const [ownerName, setOwnerName] = useState("");
  const [locationText, setLocationText] = useState(location?.location_label || "");
  const [ageYears, setAgeYears] = useState("3");
  const [ageMonths, setAgeMonths] = useState("0");
  const [sex, setSex] = useState("Female");
  const [lactation, setLactation] = useState("Not applicable");
  const [vaccination, setVaccination] = useState("FMD vaccinated");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [qr, setQr] = useState(null);
  const savedRef = useRef("");

  const species = result?.predicted_species || result?.species || "Cattle";
  const breed = (result?.predicted_breed || result?.top_breed || "Unknown").replace(/_/g, " ");
  const confidence = Math.round((Number(result?.confidence) || 0) * 100);
  const level = result?.confidence_level || 3;

  const verification = useMemo(
    () => ({
      tag,
      species,
      breed,
      confidence,
      level,
      model: result?.model_name || result?.model_used || "unknown",
      ts: new Date().toISOString(),
    }),
    [tag, species, breed, confidence, level, result]
  );

  useEffect(() => {
    if (!open) return undefined;
    let cancelled = false;
    getBpaQrMatrix(JSON.stringify(verification))
      .then((res) => {
        if (!cancelled) setQr(res);
      })
      .catch(() => {
        if (!cancelled) setQr(null);
      });
    return () => {
      cancelled = true;
    };
  }, [open, verification]);

  if (!open) return null;

  const validateTag = () => {
    if (!/^\d{12}$/.test(tag)) {
      setNotice("The BPA ear-tag UID must be exactly 12 digits.");
      return false;
    }
    return true;
  };

  const registration = () => ({
    tag,
    ownerName,
    village: locationText,
    ageYears: Number(ageYears) || 0,
    ageMonths: Number(ageMonths) || 0,
    sex,
    lactation,
    vaccination,
    species,
    breed,
    confidence,
    confidenceLevel: level,
    gps: location || null,
    registeredAt: new Date().toISOString(),
    flwId: "FLW-TN-SALEM-01",
  });

  const saveToHerd = () => {
    if (!validateTag()) return;
    const reg = registration();
    if (savedRef.current === tag) {
      setNotice(`Registration ${tag} is already saved in the herd session.`);
      return reg;
    }
    const entry = makeSessionEntry({
      result: { ...result, predicted_species: species, predicted_breed: breed },
      quality,
      thumbnail: null,
      filename: `bpa_${tag}.jpg`,
    });
    entry.bpa = reg;
    entry.id = `bpa_${tag}_${Date.now()}`;
    addSessionEntry(entry);
    savedRef.current = tag;
    setNotice(`Registration ${tag} saved to the herd session.`);
    return reg;
  };

  const drawQr = (ctx, matrix, x, y, size) => {
    if (!matrix || !Array.isArray(matrix) || !matrix.length) return false;
    const n = matrix.length;
    const cell = Math.max(1, Math.floor(size / n));
    const rendered = cell * n;
    const offsetX = x + Math.floor((size - rendered) / 2);
    const offsetY = y + Math.floor((size - rendered) / 2);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(x, y, size, size);
    ctx.fillStyle = "#0b0f19";
    for (let row = 0; row < n; row += 1) {
      if (!Array.isArray(matrix[row])) continue;
      for (let col = 0; col < n; col += 1) {
        if (matrix[row][col]) {
          ctx.fillRect(offsetX + col * cell, offsetY + row * cell, cell, cell);
        }
      }
    }
    return true;
  };

  const downloadPng = async () => {
    if (!validateTag()) return;
    setBusy(true);
    try {
      saveToHerd();
      const reg = registration();
      const canvas = document.createElement("canvas");
      canvas.width = 1080;
      canvas.height = 1500;
      const ctx = canvas.getContext("2d");
      const bg = ctx.createLinearGradient(0, 0, 1080, 1500);
      bg.addColorStop(0, "#06261c");
      bg.addColorStop(1, "#0b1220");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, 1080, 1400);
      ctx.fillStyle = "#10b981";
      ctx.fillRect(0, 0, 1080, 14);

      ctx.fillStyle = "#f8fafc";
      ctx.font = "800 44px Inter, system-ui, sans-serif";
      ctx.fillText("Bharat Pashudhan", 64, 92);
      ctx.font = "600 30px Inter, system-ui, sans-serif";
      ctx.fillStyle = "#a7f3d0";
      ctx.fillText("Digital Pashu Health & Breed Card", 64, 136);

      if (modelInputUrl) {
        try {
          const image = await loadImage(modelInputUrl);
          ctx.save();
          roundRect(ctx, 64, 180, 400, 400, 30);
          ctx.clip();
          const scale = Math.max(400 / image.width, 400 / image.height);
          const sw = 400 / scale;
          const sh = 400 / scale;
          ctx.drawImage(image, (image.width - sw) / 2, (image.height - sh) / 2, sw, sh, 64, 180, 400, 400);
          ctx.restore();
        } catch {
          /* photo is optional */
        }
      }

      let y = 210;
      const rows = [
        ["Ear Tag UID", reg.tag],
        ["Species", reg.species],
        ["Verified Breed", reg.breed],
        ["AI Confidence", `${reg.confidence}% · Level ${reg.confidenceLevel}`],
        ["Owner", reg.ownerName || "—"],
        ["Village / Taluk", reg.village || "—"],
        ["Age", `${reg.ageYears} yr ${reg.ageMonths} mo`],
        ["Sex", reg.sex],
        ["Lactation", reg.lactation],
        ["Vaccination", reg.vaccination],
        ["FLW ID", reg.flwId],
        ["Registered", new Date(reg.registeredAt).toLocaleString()],
      ];
      for (const [label, value] of rows) {
        ctx.fillStyle = "#94a3b8";
        ctx.font = "500 22px Inter, system-ui, sans-serif";
        ctx.fillText(String(label).toUpperCase(), 500, y);
        ctx.fillStyle = "#f1f5f9";
        ctx.font = "700 28px Inter, system-ui, sans-serif";
        ctx.fillText(String(value).slice(0, 34), 500, y + 34);
        y += 80;
      }

      if (qr?.matrix) {
        drawQr(ctx, qr.matrix, 790, 1160, 210);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "500 18px Inter, system-ui, sans-serif";
        ctx.fillText("Scan to verify tag + breed JSON", 790, 1392);
      }

      ctx.fillStyle = "#64748b";
      ctx.font = "500 20px Inter, system-ui, sans-serif";
      ctx.fillText("BreedID Livestock Vision · Soft-routed 3-head EfficientNet-Lite", 64, 1418);
      ctx.fillText(`Model: ${result?.model_name || "unknown"}`, 64, 1450);
      ctx.fillText("Generated offline from the FLW device. Confirm with a veterinary officer.", 64, 1480);

      await shareOrDownloadCanvas(canvas, `bpa_${reg.tag}_${Date.now()}.png`);
      setNotice("Digital health card PNG generated.");
    } catch (err) {
      setNotice(err?.message || "Could not generate the card.");
    } finally {
      setBusy(false);
    }
  };

  const downloadOdt = async () => {
    if (!validateTag()) return;
    setBusy(true);
    try {
      saveToHerd();
      const reg = registration();
      const payload = {
        ...(result || {}),
        filename: `BPA_${reg.tag}.jpg`,
        species: reg.species,
        predicted_species: reg.species,
        species_confidence: result?.species_confidence || 0,
        top_breed: reg.breed,
        predicted_breed: reg.breed,
        top_breed_confidence: reg.confidence,
        confidence: (Number(result?.confidence) || 0),
        top5_breeds: result?.top5_breeds || [],
        bpa: reg,
      };
      const blob = await exportOdt(payload, "single");
      downloadBlob(blob, `bpa_${reg.tag}.odt`);
      setNotice("Printable ODT registration card generated.");
    } catch (err) {
      setNotice(err?.message || "Could not generate the ODT card.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="advanced-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <section className="advanced-panel bpa-modal" onClick={(event) => event.stopPropagation()}>
        <header className="advanced-head">
          <div>
            <h3>Register in BPA (Bharat Pashudhan)</h3>
            <p>National Digital Livestock Mission ear-tag registration and health card.</p>
          </div>
          <button type="button" className="camera-close-icon" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <div className="advanced-body bpa-body">
          <div className="bpa-summary">
            <div>
              <span>Species</span>
              <strong>{species}</strong>
            </div>
            <div>
              <span>Predicted breed</span>
              <strong>{breed}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>
                {confidence}% · Level {level}
              </strong>
            </div>
            <div>
              <span>Top-3</span>
              <strong>
                {(result?.top5_breeds || result?.top_k || [])
                  .slice(0, 3)
                  .map((item) => item.breed)
                  .join(" · ") || "—"}
              </strong>
            </div>
          </div>

          <label className="form-label" htmlFor="bpa-tag">
            BPA / NDLM Ear Tag UID (12 digits)
          </label>
          <div className="bpa-tag-row">
            <input
              id="bpa-tag"
              className="form-input"
              value={tag}
              readOnly={!customTag}
              onChange={(event) => setTag(event.target.value.replace(/\D/g, "").slice(0, 12))}
              maxLength={12}
            />
            <button type="button" className="btn btn-outline btn-sm" onClick={() => setTag(generateBpaTag())}>
              Regenerate
            </button>
            <label className="bpa-inline-check">
              <input type="checkbox" checked={customTag} onChange={(event) => setCustomTag(event.target.checked)} />
              Type custom
            </label>
          </div>

          <div className="bpa-form-grid">
            <label>
              Owner name
              <input className="form-input" value={ownerName} onChange={(e) => setOwnerName(e.target.value)} />
            </label>
            <label>
              Village / Taluk
              <input className="form-input" value={locationText} onChange={(e) => setLocationText(e.target.value)} />
            </label>
            <label>
              Age (years)
              <input className="form-input" type="number" min="0" value={ageYears} onChange={(e) => setAgeYears(e.target.value)} />
            </label>
            <label>
              Age (months)
              <input className="form-input" type="number" min="0" max="11" value={ageMonths} onChange={(e) => setAgeMonths(e.target.value)} />
            </label>
            <label>
              Sex
              <select className="form-select" value={sex} onChange={(e) => setSex(e.target.value)}>
                {SEXES.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
            <label>
              Lactation status
              <select className="form-select" value={lactation} onChange={(e) => setLactation(e.target.value)}>
                {LACTATION.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
            <label>
              Vaccination
              <select className="form-select" value={vaccination} onChange={(e) => setVaccination(e.target.value)}>
                {VACCINATION.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </label>
            <label>
              FLW ID
              <input className="form-input" value="FLW-TN-SALEM-01" readOnly />
            </label>
          </div>

          <p className="bpa-gps">
            GPS: {location?.gps_lat != null && location?.gps_lng != null
              ? `${Number(location.gps_lat).toFixed(5)}, ${Number(location.gps_lng).toFixed(5)}`
              : "not captured"} ·{" "}
            {new Date().toLocaleString()}
          </p>

          {notice && <div className="bpa-notice">{notice}</div>}

          <div className="cropper-actions bpa-actions">
            <button type="button" className="btn btn-outline" onClick={onClose} disabled={busy}>
              Cancel
            </button>
            <button type="button" className="btn btn-secondary" onClick={saveToHerd} disabled={busy}>
              Save to Herd Session
            </button>
            <button type="button" className="btn btn-outline" onClick={downloadOdt} disabled={busy}>
              Download ODT
            </button>
            <button type="button" className="btn btn-primary" onClick={downloadPng} disabled={busy}>
              {busy ? "Generating…" : "Download PNG Health Card"}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
