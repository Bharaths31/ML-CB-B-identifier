#!/usr/bin/env python3
"""
🐄 Breed Classifier — Model Tester GUI
========================================

Standalone GUI for testing exported models on individual images.
Upload an image (PNG/JPG/JPEG), select a model checkpoint, and get
the predicted species + breed with confidence percentages.

Usage:
    python test_model.py                          # Auto-detect best checkpoint
    python test_model.py --checkpoint path/to.pt  # Use specific checkpoint
    python test_model.py --port 8501              # Custom port

Opens a browser window with the GUI at http://localhost:8501
"""

import argparse
import base64
import glob
import io
import json
import os
import sys
import threading
import time
import webbrowser

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
#  Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (CHECKPOINT_DIR, EXPORT_DIR, IMAGE_SIZE,
                        PORTABLE_EXPORT_DIR, SPLIT_DIR)
from src.model import BreedClassifier

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

SPECIES_LABELS = {0: "Cattle", 1: "Buffalo"}
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
#  Model management
# ---------------------------------------------------------------------------

class ModelManager:
    """Discovers, loads, and caches models for prediction."""

    def __init__(self):
        self.models = {}          # name -> {"path": ..., "model": ..., "backbone": ...}
        self.class_maps = {}      # "cattle" / "buffalo" -> {breed: idx}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_class_maps()
        self._discover_models()

    def _load_class_maps(self):
        """Load breed label maps from data/splits/."""
        for species in ("cattle", "buffalo"):
            path = os.path.join(SPLIT_DIR, f"{species}_classes.json")
            if os.path.exists(path):
                with open(path) as f:
                    self.class_maps[species] = json.load(f)

        # Also check portable export directories
        if not self.class_maps:
            for portable_dir in glob.glob(os.path.join(PORTABLE_EXPORT_DIR, "*")):
                for species in ("cattle", "buffalo"):
                    path = os.path.join(portable_dir, f"{species}_classes.json")
                    if os.path.exists(path):
                        with open(path) as f:
                            self.class_maps[species] = json.load(f)
                if len(self.class_maps) == 2:
                    break

    def _discover_models(self):
        """Find all available checkpoints."""
        found = {}

        # 1. Checkpoints directory
        if os.path.isdir(CHECKPOINT_DIR):
            for f in sorted(os.listdir(CHECKPOINT_DIR)):
                if f.endswith(".pt"):
                    path = os.path.join(CHECKPOINT_DIR, f)
                    name = f.replace(".pt", "")
                    backbone = "lite2" if "lite2" in f else ("lite4" if "lite4" in f else "lite2")
                    found[name] = {"path": path, "backbone": backbone, "model": None}

        # 2. Portable exports
        if os.path.isdir(PORTABLE_EXPORT_DIR):
            for d in sorted(os.listdir(PORTABLE_EXPORT_DIR)):
                model_pt = os.path.join(PORTABLE_EXPORT_DIR, d, "model.pt")
                if os.path.exists(model_pt):
                    backbone = "lite2" if "lite2" in d else ("lite4" if "lite4" in d else "lite2")
                    found[f"portable/{d}"] = {"path": model_pt, "backbone": backbone, "model": None}

        self.models = found

    def list_models(self):
        """Return list of available model names with metadata."""
        result = []
        for name, info in self.models.items():
            size_mb = os.path.getsize(info["path"]) / (1024 * 1024)
            result.append({
                "name": name,
                "path": info["path"],
                "backbone": info["backbone"],
                "size_mb": round(size_mb, 1),
            })
        return result

    def _load_model(self, name):
        """Load a model checkpoint into memory."""
        info = self.models[name]
        if info["model"] is not None:
            return info["model"]

        backbone = info["backbone"]
        model = BreedClassifier(backbone=backbone)
        ckpt = torch.load(info["path"], map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        # Filter out QAT observer keys that don't exist in the base model
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state.items() if k in model_keys}
        model.load_state_dict(filtered, strict=False)
        model.to(self.device)
        model.eval()
        info["model"] = model
        return model

    @torch.no_grad()
    def predict(self, image_bytes, model_name):
        """Run prediction on raw image bytes. Returns result dict."""
        if model_name not in self.models:
            return {"error": f"Model '{model_name}' not found"}

        # Load and preprocess image
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {"error": f"Invalid image: {e}"}

        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)
        model = self._load_model(model_name)

        # Forward pass
        out = model(tensor)
        binary_logits = out["binary"][0]
        cattle_logits = out["cattle"][0]
        buffalo_logits = out["buffalo"][0]

        # Species prediction
        binary_probs = F.softmax(binary_logits, dim=0)
        species_idx = binary_probs.argmax().item()
        species_name = SPECIES_LABELS[species_idx]
        species_conf = binary_probs[species_idx].item() * 100

        # Breed prediction based on species
        if species_idx == 0:  # Cattle
            breed_probs = F.softmax(cattle_logits, dim=0)
            class_map = self.class_maps.get("cattle", {})
        else:  # Buffalo
            breed_probs = F.softmax(buffalo_logits, dim=0)
            class_map = self.class_maps.get("buffalo", {})

        # Invert class map: idx -> breed_name
        idx_to_breed = {v: k for k, v in class_map.items()}

        # Top-5 breed predictions
        top5_probs, top5_idxs = breed_probs.topk(min(5, len(breed_probs)))
        top5 = []
        for prob, idx in zip(top5_probs.tolist(), top5_idxs.tolist()):
            breed = idx_to_breed.get(idx, f"class_{idx}")
            # Clean up breed name for display
            display_name = breed.replace("_", " ").title()
            top5.append({
                "breed": display_name,
                "confidence": round(prob * 100, 2),
            })

        return {
            "species": species_name,
            "species_confidence": round(species_conf, 2),
            "top_breed": top5[0]["breed"] if top5 else "Unknown",
            "top_breed_confidence": top5[0]["confidence"] if top5 else 0,
            "top5_breeds": top5,
            "model_used": model_name,
        }


# ---------------------------------------------------------------------------
#  HTTP Server
# ---------------------------------------------------------------------------

def build_html():
    """Return the complete single-page GUI HTML."""
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🐄 Breed Classifier — Model Tester</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0a0e17;
      --surface: #111827;
      --surface2: #1e293b;
      --border: #2d3a4f;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --accent: #6366f1;
      --accent-light: #818cf8;
      --green: #22c55e;
      --green-bg: rgba(34,197,94,0.1);
      --amber: #f59e0b;
      --amber-bg: rgba(245,158,11,0.1);
      --red: #ef4444;
      --radius: 16px;
      --radius-sm: 10px;
    }

    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* --- Animated background --- */
    body::before {
      content: '';
      position: fixed;
      top: -50%; left: -50%;
      width: 200%; height: 200%;
      background: radial-gradient(ellipse at 30% 20%, rgba(99,102,241,0.08) 0%, transparent 50%),
                  radial-gradient(ellipse at 80% 80%, rgba(34,197,94,0.06) 0%, transparent 50%);
      animation: drift 20s ease-in-out infinite;
      z-index: -1;
    }
    @keyframes drift {
      0%,100% { transform: translate(0,0); }
      50% { transform: translate(-3%,3%); }
    }

    /* --- Header --- */
    .header {
      text-align: center;
      padding: 48px 24px 24px;
    }
    .header h1 {
      font-size: 2.2rem;
      font-weight: 800;
      background: linear-gradient(135deg, var(--accent-light), var(--green));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 6px;
    }
    .header p {
      color: var(--text-muted);
      font-size: 0.95rem;
      font-weight: 300;
    }

    /* --- Layout --- */
    .container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 0 24px 60px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    @media (max-width: 768px) {
      .container { grid-template-columns: 1fr; }
    }

    /* --- Card --- */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 28px;
      transition: border-color 0.3s;
    }
    .card:hover { border-color: var(--accent); }
    .card h2 {
      font-size: 1.1rem;
      font-weight: 600;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* --- Model Selector --- */
    .model-selector {
      background: var(--surface2);
      border-radius: var(--radius-sm);
      padding: 14px 16px;
      margin-bottom: 20px;
    }
    .model-selector label {
      display: block;
      font-size: 0.8rem;
      font-weight: 500;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }
    .model-selector select {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 0.9rem;
      font-family: inherit;
      cursor: pointer;
      outline: none;
      transition: border-color 0.2s;
    }
    .model-selector select:focus { border-color: var(--accent); }
    .model-meta {
      margin-top: 8px;
      font-size: 0.78rem;
      color: var(--text-muted);
    }

    /* --- Dropzone --- */
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: var(--radius-sm);
      padding: 40px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s;
      position: relative;
    }
    .dropzone:hover, .dropzone.drag-over {
      border-color: var(--accent);
      background: rgba(99,102,241,0.05);
    }
    .dropzone .icon {
      font-size: 3rem;
      margin-bottom: 12px;
      display: block;
    }
    .dropzone p {
      color: var(--text-muted);
      font-size: 0.88rem;
      line-height: 1.6;
    }
    .dropzone .formats {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 6px;
      opacity: 0.7;
    }

    /* --- Image Preview --- */
    .preview-container {
      position: relative;
      border-radius: var(--radius-sm);
      overflow: hidden;
      margin-bottom: 16px;
    }
    .preview-container img {
      width: 100%;
      height: 260px;
      object-fit: cover;
      display: block;
      border-radius: var(--radius-sm);
    }
    .clear-btn {
      position: absolute;
      top: 10px;
      right: 10px;
      background: rgba(0,0,0,0.6);
      backdrop-filter: blur(8px);
      border: none;
      color: white;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      cursor: pointer;
      font-size: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.2s;
    }
    .clear-btn:hover { background: var(--red); }

    /* --- Predict Button --- */
    .predict-btn {
      width: 100%;
      padding: 14px;
      border: none;
      border-radius: var(--radius-sm);
      background: linear-gradient(135deg, var(--accent), #8b5cf6);
      color: white;
      font-size: 1rem;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      transition: all 0.3s;
      position: relative;
      overflow: hidden;
    }
    .predict-btn:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 8px 25px rgba(99,102,241,0.3);
    }
    .predict-btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }
    .predict-btn.loading {
      pointer-events: none;
    }
    .predict-btn.loading::after {
      content: '';
      position: absolute;
      top: 0; left: -100%;
      width: 200%; height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
      animation: shimmer 1.5s infinite;
    }
    @keyframes shimmer {
      100% { transform: translateX(100%); }
    }

    /* --- Results --- */
    .results { display: none; }
    .results.visible { display: block; animation: fadeUp 0.5s ease; }
    @keyframes fadeUp {
      from { opacity:0; transform:translateY(12px); }
      to { opacity:1; transform:translateY(0); }
    }

    .result-hero {
      background: var(--surface2);
      border-radius: var(--radius-sm);
      padding: 24px;
      text-align: center;
      margin-bottom: 16px;
      border: 1px solid var(--border);
    }
    .result-hero .species-badge {
      display: inline-block;
      padding: 6px 16px;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 12px;
    }
    .result-hero .species-badge.cattle {
      background: var(--amber-bg);
      color: var(--amber);
      border: 1px solid rgba(245,158,11,0.3);
    }
    .result-hero .species-badge.buffalo {
      background: var(--green-bg);
      color: var(--green);
      border: 1px solid rgba(34,197,94,0.3);
    }
    .result-hero .breed-name {
      font-size: 1.8rem;
      font-weight: 800;
      margin-bottom: 4px;
      line-height: 1.2;
    }
    .result-hero .confidence {
      font-size: 2.2rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--green), #4ade80);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .result-hero .conf-label {
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 2px;
    }

    /* --- Top-5 Bar Chart --- */
    .top5-list { list-style: none; }
    .top5-item {
      display: grid;
      grid-template-columns: 1fr 60px;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
    }
    .top5-bar-wrap {
      position: relative;
      height: 32px;
      background: var(--surface2);
      border-radius: 6px;
      overflow: hidden;
    }
    .top5-bar {
      height: 100%;
      border-radius: 6px;
      background: linear-gradient(90deg, var(--accent), var(--accent-light));
      transition: width 0.8s cubic-bezier(0.22,1,0.36,1);
    }
    .top5-bar-wrap .breed-label {
      position: absolute;
      top: 50%; left: 12px;
      transform: translateY(-50%);
      font-size: 0.82rem;
      font-weight: 500;
      color: white;
      white-space: nowrap;
      text-shadow: 0 1px 3px rgba(0,0,0,0.5);
    }
    .top5-pct {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      text-align: right;
    }

    /* --- Info footer --- */
    .info-footer {
      margin-top: 16px;
      padding: 12px 16px;
      background: var(--surface2);
      border-radius: 8px;
      font-size: 0.78rem;
      color: var(--text-muted);
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }
    .info-footer span { display: flex; align-items: center; gap: 4px; }

    /* --- Status chip --- */
    .status-bar {
      text-align: center;
      padding: 8px;
      font-size: 0.78rem;
      color: var(--text-muted);
    }
    .status-bar .chip {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 12px;
      background: var(--surface);
      border: 1px solid var(--border);
      margin: 0 4px;
    }
    .chip.gpu { border-color: var(--green); color: var(--green); }
    .chip.cpu { border-color: var(--amber); color: var(--amber); }

    .hidden { display: none !important; }
  </style>
</head>
<body>

<div class="header">
  <h1>🐄 Breed Classifier — Model Tester</h1>
  <p>Upload an image to identify cattle and buffalo breeds with confidence scores</p>
</div>

<div class="status-bar" id="status-bar"></div>

<div class="container">
  <!-- LEFT COLUMN: Upload & Model Selection -->
  <div>
    <div class="card">
      <h2>📸 Image Upload</h2>

      <div class="dropzone" id="dropzone">
        <span class="icon">📷</span>
        <p>Drag & drop an image here<br>or click to browse</p>
        <div class="formats">Supports: PNG, JPG, JPEG, BMP, WebP</div>
        <input type="file" id="file-input" accept=".png,.jpg,.jpeg,.bmp,.webp" hidden>
      </div>

      <div class="preview-container hidden" id="preview-wrap">
        <img id="preview-img" alt="Preview">
        <button class="clear-btn" id="clear-btn" title="Clear image">✕</button>
      </div>

      <div class="model-selector">
        <label for="model-select">Select Model</label>
        <select id="model-select"></select>
        <div class="model-meta" id="model-meta"></div>
      </div>

      <button class="predict-btn" id="predict-btn" disabled>
        🔍 Analyze Breed
      </button>
    </div>
  </div>

  <!-- RIGHT COLUMN: Results -->
  <div>
    <div class="card">
      <h2>📊 Prediction Results</h2>

      <div class="results" id="results">
        <div class="result-hero" id="result-hero">
          <div class="species-badge" id="species-badge"></div>
          <div class="breed-name" id="breed-name"></div>
          <div class="confidence" id="breed-conf"></div>
          <div class="conf-label">Breed Confidence</div>
        </div>

        <h2 style="margin-top:20px">🏆 Top 5 Predictions</h2>
        <ul class="top5-list" id="top5-list"></ul>

        <div class="info-footer" id="info-footer"></div>
      </div>

      <div id="placeholder" style="text-align:center;padding:60px 20px;color:var(--text-muted)">
        <span style="font-size:3rem;display:block;margin-bottom:12px">🔬</span>
        <p>Upload an image and click <strong>Analyze Breed</strong><br>to see predictions here</p>
      </div>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let currentFile = null;

// --- Init: load models list ---
fetch('/api/models').then(r => r.json()).then(data => {
  const sel = $('#model-select');
  const meta = $('#model-meta');
  data.models.forEach((m, i) => {
    const opt = document.createElement('option');
    opt.value = m.name;
    opt.textContent = `${m.name} (${m.size_mb} MB)`;
    if (m.name.includes('phase2')) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => {
    const m = data.models.find(x => x.name === sel.value);
    meta.textContent = m ? `Backbone: ${m.backbone} · ${m.size_mb} MB · ${m.path}` : '';
  });
  sel.dispatchEvent(new Event('change'));

  // Status bar
  const sb = $('#status-bar');
  const deviceChip = data.device === 'cuda'
    ? `<span class="chip gpu">🟢 GPU (CUDA)</span>`
    : `<span class="chip cpu">🟡 CPU</span>`;
  sb.innerHTML = `${deviceChip} <span class="chip">${data.models.length} models available</span>`;
});

// --- Dropzone ---
const dz = $('#dropzone');
const fi = $('#file-input');
const pw = $('#preview-wrap');
const pi = $('#preview-img');
const pb = $('#predict-btn');

dz.addEventListener('click', () => fi.click());
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fi.addEventListener('change', () => { if (fi.files.length) handleFile(fi.files[0]); });

$('#clear-btn').addEventListener('click', () => {
  currentFile = null;
  pw.classList.add('hidden');
  dz.classList.remove('hidden');
  pb.disabled = true;
  $('#results').classList.remove('visible');
  $('#placeholder').classList.remove('hidden');
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    pi.src = e.target.result;
    pw.classList.remove('hidden');
    dz.classList.add('hidden');
    pb.disabled = false;
  };
  reader.readAsDataURL(file);
}

// --- Predict ---
pb.addEventListener('click', async () => {
  if (!currentFile) return;
  pb.disabled = true;
  pb.classList.add('loading');
  pb.textContent = '⏳ Analyzing...';

  const form = new FormData();
  form.append('image', currentFile);
  form.append('model', $('#model-select').value);

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    showResults(data);
  } catch (e) {
    alert('Prediction failed: ' + e.message);
  } finally {
    pb.disabled = false;
    pb.classList.remove('loading');
    pb.textContent = '🔍 Analyze Breed';
  }
});

function showResults(data) {
  $('#placeholder').classList.add('hidden');
  const r = $('#results');
  r.classList.add('visible');

  // Hero
  const badge = $('#species-badge');
  badge.textContent = data.species;
  badge.className = 'species-badge ' + data.species.toLowerCase();

  $('#breed-name').textContent = data.top_breed;
  $('#breed-conf').textContent = data.top_breed_confidence.toFixed(1) + '%';

  // Top 5
  const list = $('#top5-list');
  list.innerHTML = '';
  const maxConf = Math.max(...data.top5_breeds.map(b => b.confidence), 1);
  data.top5_breeds.forEach((b, i) => {
    const li = document.createElement('li');
    li.className = 'top5-item';
    const barWidth = Math.max(8, (b.confidence / maxConf) * 100);
    li.innerHTML = `
      <div class="top5-bar-wrap">
        <div class="top5-bar" style="width:0%"></div>
        <span class="breed-label">${b.breed}</span>
      </div>
      <span class="top5-pct">${b.confidence.toFixed(1)}%</span>
    `;
    list.appendChild(li);
    // Animate bar
    requestAnimationFrame(() => {
      setTimeout(() => {
        li.querySelector('.top5-bar').style.width = barWidth + '%';
      }, i * 100);
    });
  });

  // Info footer
  $('#info-footer').innerHTML = `
    <span>🤖 Model: ${data.model_used}</span>
    <span>🐄 Species: ${data.species} (${data.species_confidence.toFixed(1)}%)</span>
  `;
}
</script>
</body>
</html>"""


def run_server(manager, port=8501):
    """Run a minimal HTTP server with the GUI and prediction API."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import email.parser
    import re

    html_content = build_html()

    def _parse_multipart(handler):
        """Parse multipart/form-data without the deprecated cgi module."""
        content_type = handler.headers.get("Content-Type", "")
        # Extract boundary
        m = re.search(r'boundary=([^\s;]+)', content_type)
        if not m:
            return {}, {}
        boundary = m.group(1).encode()
        content_length = int(handler.headers.get("Content-Length", 0))
        body = handler.rfile.read(content_length)

        parts_data = {}   # name -> bytes
        parts_text = {}   # name -> str

        # Split by boundary
        chunks = body.split(b"--" + boundary)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or chunk == b"--":
                continue
            # Split headers from body
            sep = chunk.find(b"\r\n\r\n")
            if sep < 0:
                continue
            header_bytes = chunk[:sep]
            part_body = chunk[sep+4:]
            # Remove trailing \r\n
            if part_body.endswith(b"\r\n"):
                part_body = part_body[:-2]

            # Parse disposition
            header_str = header_bytes.decode("utf-8", errors="replace")
            name_match = re.search(r'name="([^"]+)"', header_str)
            if not name_match:
                continue
            name = name_match.group(1)

            filename_match = re.search(r'filename="([^"]*)"', header_str)
            if filename_match:
                parts_data[name] = part_body  # binary file
            else:
                parts_text[name] = part_body.decode("utf-8", errors="replace")

        return parts_data, parts_text

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress default access logs

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._respond(200, "text/html", html_content.encode())
            elif self.path == "/api/models":
                models = manager.list_models()
                device = "cuda" if manager.device.type == "cuda" else "cpu"
                body = json.dumps({"models": models, "device": device})
                self._respond(200, "application/json", body.encode())
            else:
                self._respond(404, "text/plain", b"Not Found")

        def do_POST(self):
            if self.path == "/api/predict":
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self._respond(400, "application/json",
                                  json.dumps({"error": "Expected multipart/form-data"}).encode())
                    return

                parts_data, parts_text = _parse_multipart(self)
                image_bytes = parts_data.get("image")
                model_name = parts_text.get("model", "")

                if not image_bytes:
                    self._respond(400, "application/json",
                                  json.dumps({"error": "No image provided"}).encode())
                    return

                result = manager.predict(image_bytes, model_name)
                self._respond(200, "application/json", json.dumps(result).encode())
            else:
                self._respond(404, "text/plain", b"Not Found")

        def _respond(self, code, content_type, body):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n  🌐 Model Tester GUI running at http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")
    server.serve_forever()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="🐄 Breed Classifier — Model Tester GUI")
    parser.add_argument("--port", type=int, default=8501,
                        help="port to serve on (default: 8501)")
    parser.add_argument("--no-browser", action="store_true",
                        help="don't auto-open browser")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🐄 Breed Classifier — Model Tester GUI")
    print("=" * 60)

    manager = ModelManager()
    models = manager.list_models()

    if not models:
        print("\n  ❌ No models found!")
        print("  Train a model first:")
        print("    python local_train.py --smoke-test")
        print("    python local_train.py --quarter-data")
        return 1

    print(f"\n  Device: {manager.device}")
    print(f"  Models found: {len(models)}")
    for m in models:
        print(f"    • {m['name']} ({m['backbone']}, {m['size_mb']} MB)")

    if not manager.class_maps:
        print("\n  ⚠️  Warning: No class maps found in data/splits/")
        print("  Breed names will show as class indices.")

    # Auto-open browser
    if not args.no_browser:
        url = f"http://localhost:{args.port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        run_server(manager, port=args.port)
    except KeyboardInterrupt:
        print("\n  Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
