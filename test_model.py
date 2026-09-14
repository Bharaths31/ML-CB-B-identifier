#!/usr/bin/env python3
"""
🐄 Breed Classifier — Model Tester GUI
========================================

Standalone GUI for testing exported models on individual or batch images.
Upload image(s) (PNG/JPG/JPEG), select a model checkpoint, and get
the predicted species + breed with confidence percentages.
Supports exporting results to ODT format.

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
import re
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
                if f.endswith(".pt") or f.endswith(".onnx"):
                    path = os.path.join(CHECKPOINT_DIR, f)
                    name = f.replace(".pt", "").replace(".onnx", " (ONNX)")
                    backbone = "lite2" if "lite2" in f else ("lite4" if "lite4" in f else "lite2")
                    mtype = "onnx" if f.endswith(".onnx") else "pt"
                    found[name] = {"path": path, "backbone": backbone, "model": None, "type": mtype}

        # 2. Portable exports
        if os.path.isdir(PORTABLE_EXPORT_DIR):
            for d in sorted(os.listdir(PORTABLE_EXPORT_DIR)):
                model_pt = os.path.join(PORTABLE_EXPORT_DIR, d, "model.pt")
                if os.path.exists(model_pt):
                    backbone = "lite2" if "lite2" in d else ("lite4" if "lite4" in d else "lite2")
                    found[f"portable/{d}"] = {"path": model_pt, "backbone": backbone, "model": None, "type": "pt"}

        # 3. ONNX Exports directory
        if os.path.isdir(EXPORT_DIR):
            for f in sorted(os.listdir(EXPORT_DIR)):
                if f.endswith(".onnx"):
                    path = os.path.join(EXPORT_DIR, f)
                    name = f"export/{f.replace('.onnx', '')} (ONNX)"
                    backbone = "lite2" if "lite2" in f else ("lite4" if "lite4" in f else "lite2")
                    found[name] = {"path": path, "backbone": backbone, "model": None, "type": "onnx"}

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

        if info.get("type") == "onnx":
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                session = ort.InferenceSession(info["path"], providers=providers)
                info["model"] = session
                return session
            except ImportError:
                raise RuntimeError("onnxruntime is not installed. Please `pip install onnxruntime` to test ONNX models.")

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

        tensor = TRANSFORM(img).unsqueeze(0)
        info = self.models[model_name]

        try:
            model = self._load_model(model_name)
        except Exception as e:
            return {"error": str(e)}

        if info.get("type") == "onnx":
            # ONNX Inference
            input_name = model.get_inputs()[0].name
            ort_outs = model.run(None, {input_name: tensor.numpy()})
            out_names = [x.name for x in model.get_outputs()]
            out_dict = dict(zip(out_names, ort_outs))
            binary_logits = torch.tensor(out_dict.get("binary", ort_outs[0])[0])
            cattle_logits = torch.tensor(out_dict.get("cattle", ort_outs[1])[0])
            buffalo_logits = torch.tensor(out_dict.get("buffalo", ort_outs[2])[0])
        else:
            # PyTorch Inference
            tensor = tensor.to(self.device)
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
#  ODT Export
# ---------------------------------------------------------------------------

def generate_odt_report(results, mode="single"):
    """Generate an ODT report from prediction results.

    Args:
        results: single result dict or list of result dicts (batch).
        mode: "single" or "batch".

    Returns:
        bytes: ODT file content.
    """
    from odf.opendocument import OpenDocumentText
    from odf.style import Style, TextProperties, ParagraphProperties, TableColumnProperties, TableCellProperties
    from odf.text import P, H
    from odf.table import Table, TableColumn, TableRow, TableCell

    doc = OpenDocumentText()

    # --- Define styles ---
    title_style = Style(name="Title", family="paragraph")
    title_style.addElement(TextProperties(attributes={
        "fontsize": "20pt", "fontweight": "bold", "color": "#1a1a2e"
    }))
    title_style.addElement(ParagraphProperties(attributes={
        "marginbottom": "0.3cm", "margintop": "0.3cm"
    }))
    doc.styles.addElement(title_style)

    heading_style = Style(name="Heading", family="paragraph")
    heading_style.addElement(TextProperties(attributes={
        "fontsize": "14pt", "fontweight": "bold", "color": "#16213e"
    }))
    heading_style.addElement(ParagraphProperties(attributes={
        "marginbottom": "0.2cm", "margintop": "0.4cm"
    }))
    doc.styles.addElement(heading_style)

    subheading_style = Style(name="SubHeading", family="paragraph")
    subheading_style.addElement(TextProperties(attributes={
        "fontsize": "12pt", "fontweight": "bold", "color": "#0f3460"
    }))
    subheading_style.addElement(ParagraphProperties(attributes={
        "marginbottom": "0.15cm", "margintop": "0.3cm"
    }))
    doc.styles.addElement(subheading_style)

    body_style = Style(name="Body", family="paragraph")
    body_style.addElement(TextProperties(attributes={
        "fontsize": "11pt", "color": "#333333"
    }))
    body_style.addElement(ParagraphProperties(attributes={
        "marginbottom": "0.1cm"
    }))
    doc.styles.addElement(body_style)

    meta_style = Style(name="Meta", family="paragraph")
    meta_style.addElement(TextProperties(attributes={
        "fontsize": "9pt", "fontstyle": "italic", "color": "#666666"
    }))
    meta_style.addElement(ParagraphProperties(attributes={
        "marginbottom": "0.1cm"
    }))
    doc.styles.addElement(meta_style)

    # Table styles
    table_style = Style(name="TableStyle", family="table")
    doc.automaticstyles.addElement(table_style)

    col_wide = Style(name="ColWide", family="table-column")
    col_wide.addElement(TableColumnProperties(attributes={"columnwidth": "8cm"}))
    doc.automaticstyles.addElement(col_wide)

    col_narrow = Style(name="ColNarrow", family="table-column")
    col_narrow.addElement(TableColumnProperties(attributes={"columnwidth": "4cm"}))
    doc.automaticstyles.addElement(col_narrow)

    header_cell_style = Style(name="HeaderCell", family="table-cell")
    header_cell_style.addElement(TableCellProperties(attributes={
        "backgroundcolor": "#1a1a2e", "padding": "0.15cm",
        "borderbottom": "0.5pt solid #333333"
    }))
    doc.automaticstyles.addElement(header_cell_style)

    header_text_style = Style(name="HeaderText", family="paragraph")
    header_text_style.addElement(TextProperties(attributes={
        "fontsize": "10pt", "fontweight": "bold", "color": "#ffffff"
    }))
    doc.automaticstyles.addElement(header_text_style)

    cell_style = Style(name="DataCell", family="table-cell")
    cell_style.addElement(TableCellProperties(attributes={
        "padding": "0.1cm",
        "borderbottom": "0.5pt solid #cccccc"
    }))
    doc.automaticstyles.addElement(cell_style)

    cell_text_style = Style(name="CellText", family="paragraph")
    cell_text_style.addElement(TextProperties(attributes={
        "fontsize": "10pt", "color": "#333333"
    }))
    doc.automaticstyles.addElement(cell_text_style)

    # --- Document content ---
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Title
    p = P(stylename=title_style, text="🐄 Cattle & Buffalo Breed Classifier — Test Report")
    doc.text.addElement(p)

    p = P(stylename=meta_style, text=f"Generated: {timestamp}")
    doc.text.addElement(p)

    if mode == "single":
        results_list = [results] if isinstance(results, dict) else results
    else:
        results_list = results if isinstance(results, list) else [results]

    model_name = results_list[0].get("model_used", "unknown") if results_list else "unknown"
    p = P(stylename=meta_style, text=f"Model: {model_name}")
    doc.text.addElement(p)
    p = P(stylename=meta_style, text=f"Mode: {'Single Image' if mode == 'single' else 'Batch (' + str(len(results_list)) + ' images)'}")
    doc.text.addElement(p)

    # Separator
    p = P(stylename=body_style, text="─" * 60)
    doc.text.addElement(p)

    for i, result in enumerate(results_list):
        if "error" in result:
            p = P(stylename=body_style, text=f"Error: {result['error']}")
            doc.text.addElement(p)
            continue

        filename = result.get("filename", f"Image {i+1}")

        # Image heading
        p = P(stylename=heading_style, text=f"{'Result' if mode == 'single' else f'Image {i+1}'}: {filename}")
        doc.text.addElement(p)

        # Species
        p = P(stylename=body_style, text=f"Species: {result['species']} ({result['species_confidence']:.1f}%)")
        doc.text.addElement(p)

        # Top breed
        p = P(stylename=body_style, text=f"Predicted Breed: {result['top_breed']} ({result['top_breed_confidence']:.1f}%)")
        doc.text.addElement(p)

        # Top-5 table
        p = P(stylename=subheading_style, text="Top 5 Predictions")
        doc.text.addElement(p)

        table = Table(name=f"Top5_{i}", stylename=table_style)
        table.addElement(TableColumn(stylename=col_narrow))  # Rank
        table.addElement(TableColumn(stylename=col_wide))     # Breed
        table.addElement(TableColumn(stylename=col_narrow))   # Confidence

        # Header row
        hrow = TableRow()
        for htext in ["Rank", "Breed", "Confidence"]:
            hcell = TableCell(stylename=header_cell_style)
            hcell.addElement(P(stylename=header_text_style, text=htext))
            hrow.addElement(hcell)
        table.addElement(hrow)

        # Data rows
        for rank, breed_info in enumerate(result.get("top5_breeds", []), 1):
            row = TableRow()
            for val in [str(rank), breed_info["breed"], f"{breed_info['confidence']:.1f}%"]:
                dcell = TableCell(stylename=cell_style)
                dcell.addElement(P(stylename=cell_text_style, text=val))
                row.addElement(dcell)
            table.addElement(row)

        doc.text.addElement(table)

        # Separator between images in batch mode
        if mode == "batch" and i < len(results_list) - 1:
            p = P(stylename=body_style, text="")
            doc.text.addElement(p)
            p = P(stylename=body_style, text="─" * 60)
            doc.text.addElement(p)

    # Summary section for batch mode
    if mode == "batch" and len(results_list) > 1:
        p = P(stylename=body_style, text="")
        doc.text.addElement(p)
        p = P(stylename=heading_style, text="Batch Summary")
        doc.text.addElement(p)

        # Summary table
        summary_table = Table(name="Summary", stylename=table_style)
        summary_table.addElement(TableColumn(stylename=col_wide))
        summary_table.addElement(TableColumn(stylename=col_narrow))
        summary_table.addElement(TableColumn(stylename=col_wide))
        summary_table.addElement(TableColumn(stylename=col_narrow))

        # Header
        hrow = TableRow()
        for htext in ["Filename", "Species", "Breed", "Confidence"]:
            hcell = TableCell(stylename=header_cell_style)
            hcell.addElement(P(stylename=header_text_style, text=htext))
            hrow.addElement(hcell)
        summary_table.addElement(hrow)

        for r in results_list:
            if "error" in r:
                continue
            row = TableRow()
            for val in [
                r.get("filename", "—"),
                r.get("species", "—"),
                r.get("top_breed", "—"),
                f"{r.get('top_breed_confidence', 0):.1f}%"
            ]:
                dcell = TableCell(stylename=cell_style)
                dcell.addElement(P(stylename=cell_text_style, text=val))
                row.addElement(dcell)
            summary_table.addElement(row)

        doc.text.addElement(summary_table)

        # Stats
        valid = [r for r in results_list if "error" not in r]
        cattle_count = sum(1 for r in valid if r.get("species") == "Cattle")
        buffalo_count = sum(1 for r in valid if r.get("species") == "Buffalo")
        avg_conf = sum(r.get("top_breed_confidence", 0) for r in valid) / len(valid) if valid else 0

        p = P(stylename=body_style, text="")
        doc.text.addElement(p)
        p = P(stylename=body_style, text=f"Total Images: {len(results_list)} | Cattle: {cattle_count} | Buffalo: {buffalo_count}")
        doc.text.addElement(p)
        p = P(stylename=body_style, text=f"Average Top Breed Confidence: {avg_conf:.1f}%")
        doc.text.addElement(p)

    # Write to bytes buffer
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
#  HTML GUI
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
      padding: 48px 24px 16px;
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

    /* --- Tabs --- */
    .tabs {
      display: flex;
      justify-content: center;
      gap: 4px;
      margin: 16px auto 24px;
      background: var(--surface);
      border-radius: 12px;
      padding: 4px;
      width: fit-content;
      border: 1px solid var(--border);
    }
    .tab-btn {
      padding: 10px 28px;
      border: none;
      border-radius: 9px;
      background: transparent;
      color: var(--text-muted);
      font-family: inherit;
      font-size: 0.9rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.25s;
    }
    .tab-btn:hover { color: var(--text); background: var(--surface2); }
    .tab-btn.active {
      background: linear-gradient(135deg, var(--accent), #8b5cf6);
      color: white;
      font-weight: 600;
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

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

    /* --- Filename label --- */
    .filename-label {
      background: var(--surface2);
      border-radius: 8px;
      padding: 8px 14px;
      margin-bottom: 12px;
      font-size: 0.82rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 6px;
      word-break: break-all;
    }
    .filename-label .fname { color: var(--text); font-weight: 500; }

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
    .predict-btn.loading { pointer-events: none; }
    .predict-btn.loading::after {
      content: '';
      position: absolute;
      top: 0; left: -100%;
      width: 200%; height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
      animation: shimmer 1.5s infinite;
    }
    @keyframes shimmer { 100% { transform: translateX(100%); } }

    /* --- Export Button --- */
    .export-btn {
      width: 100%;
      padding: 12px;
      border: 1px solid var(--green);
      border-radius: var(--radius-sm);
      background: transparent;
      color: var(--green);
      font-size: 0.9rem;
      font-weight: 600;
      font-family: inherit;
      cursor: pointer;
      transition: all 0.3s;
      margin-top: 12px;
    }
    .export-btn:hover:not(:disabled) {
      background: var(--green-bg);
      transform: translateY(-1px);
    }
    .export-btn:disabled { opacity: 0.4; cursor: not-allowed; }

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

    /* --- Batch results --- */
    .batch-results-wrap { max-height: 70vh; overflow-y: auto; }
    .batch-card {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 18px;
      margin-bottom: 14px;
      transition: border-color 0.2s;
    }
    .batch-card:hover { border-color: var(--accent); }
    .batch-card .bc-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .batch-card .bc-filename {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--accent-light);
      word-break: break-all;
    }
    .batch-card .bc-species {
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    .batch-card .bc-species.cattle { background: var(--amber-bg); color: var(--amber); }
    .batch-card .bc-species.buffalo { background: var(--green-bg); color: var(--green); }
    .batch-card .bc-breed {
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .batch-card .bc-conf {
      font-size: 0.85rem;
      color: var(--green);
      font-weight: 600;
    }
    .batch-card .bc-top5 {
      margin-top: 10px;
      list-style: none;
      font-size: 0.8rem;
    }
    .batch-card .bc-top5 li {
      display: flex;
      justify-content: space-between;
      padding: 3px 0;
      color: var(--text-muted);
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .batch-card .bc-top5 li:last-child { border-bottom: none; }

    /* Batch file list */
    .batch-file-list {
      max-height: 200px;
      overflow-y: auto;
      background: var(--surface2);
      border-radius: 8px;
      padding: 10px 14px;
      margin-bottom: 14px;
    }
    .batch-file-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 5px 0;
      font-size: 0.82rem;
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .batch-file-item:last-child { border-bottom: none; }
    .batch-file-item .bf-name { color: var(--text); word-break: break-all; }
    .batch-file-item .bf-size { color: var(--text-muted); white-space: nowrap; margin-left: 12px; }

    /* Progress bar */
    .progress-wrap {
      margin: 12px 0;
      height: 6px;
      background: var(--surface2);
      border-radius: 3px;
      overflow: hidden;
    }
    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--green));
      border-radius: 3px;
      transition: width 0.3s;
    }
    .progress-text {
      text-align: center;
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 4px;
    }

    /* Batch summary */
    .batch-summary {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 18px;
      margin-bottom: 14px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      text-align: center;
    }
    .batch-summary .stat-val {
      font-size: 1.6rem;
      font-weight: 800;
      background: linear-gradient(135deg, var(--accent-light), var(--green));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .batch-summary .stat-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .hidden { display: none !important; }
  </style>
</head>
<body>

<div class="header">
  <h1>🐄 Breed Classifier — Model Tester</h1>
  <p>Upload images to identify cattle and buffalo breeds with confidence scores</p>
</div>

<div class="status-bar" id="status-bar"></div>

<!-- Tab navigation -->
<div class="tabs">
  <button class="tab-btn active" data-tab="single" id="tab-single-btn">📸 Single Image</button>
  <button class="tab-btn" data-tab="batch" id="tab-batch-btn">📁 Batch Images</button>
</div>

<!-- ============================================================ -->
<!--  SINGLE IMAGE TAB                                            -->
<!-- ============================================================ -->
<div class="tab-content active" id="tab-single">
<div class="container">
  <!-- LEFT COLUMN -->
  <div>
    <div class="card">
      <h2>📸 Image Upload</h2>

      <div class="dropzone" id="dropzone">
        <span class="icon">📷</span>
        <p>Drag & drop an image here<br>or click to browse</p>
        <div class="formats">Supports: PNG, JPG, JPEG, BMP, WebP</div>
        <input type="file" id="file-input" accept=".png,.jpg,.jpeg,.bmp,.webp" hidden>
      </div>

      <div class="filename-label hidden" id="filename-label">
        📄 <span class="fname" id="filename-text"></span>
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

        <button class="export-btn" id="export-single-btn" disabled>
          📄 Export to ODT
        </button>
      </div>

      <div id="placeholder" style="text-align:center;padding:60px 20px;color:var(--text-muted)">
        <span style="font-size:3rem;display:block;margin-bottom:12px">🔬</span>
        <p>Upload an image and click <strong>Analyze Breed</strong><br>to see predictions here</p>
      </div>
    </div>
  </div>
</div>
</div>

<!-- ============================================================ -->
<!--  BATCH IMAGE TAB                                             -->
<!-- ============================================================ -->
<div class="tab-content" id="tab-batch">
<div class="container">
  <!-- LEFT COLUMN -->
  <div>
    <div class="card">
      <h2>📁 Batch Upload</h2>

      <div class="dropzone" id="batch-dropzone">
        <span class="icon">📂</span>
        <p>Drag & drop multiple images here<br>or click to browse</p>
        <div class="formats">Supports: PNG, JPG, JPEG, BMP, WebP</div>
        <input type="file" id="batch-file-input" accept=".png,.jpg,.jpeg,.bmp,.webp" multiple hidden>
      </div>

      <div class="batch-file-list hidden" id="batch-file-list"></div>

      <div class="model-selector">
        <label for="batch-model-select">Select Model</label>
        <select id="batch-model-select"></select>
        <div class="model-meta" id="batch-model-meta"></div>
      </div>

      <button class="predict-btn" id="batch-predict-btn" disabled>
        🔍 Analyze All Images
      </button>

      <div class="progress-wrap hidden" id="batch-progress-wrap">
        <div class="progress-bar" id="batch-progress-bar" style="width:0%"></div>
      </div>
      <div class="progress-text hidden" id="batch-progress-text"></div>

      <button class="predict-btn hidden" id="batch-clear-btn"
              style="background:var(--surface2);color:var(--text-muted);margin-top:10px;border:1px solid var(--border)">
        🗑️ Clear All
      </button>
    </div>
  </div>

  <!-- RIGHT COLUMN: Batch Results -->
  <div>
    <div class="card">
      <h2>📊 Batch Results</h2>

      <div class="hidden" id="batch-results">
        <div class="batch-summary" id="batch-summary"></div>
        <button class="export-btn" id="export-batch-btn">
          📄 Export All to ODT
        </button>
        <div class="batch-results-wrap" id="batch-results-list" style="margin-top:14px"></div>
      </div>

      <div id="batch-placeholder" style="text-align:center;padding:60px 20px;color:var(--text-muted)">
        <span style="font-size:3rem;display:block;margin-bottom:12px">📊</span>
        <p>Select images and click <strong>Analyze All</strong><br>to see batch results here</p>
      </div>
    </div>
  </div>
</div>
</div>

<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
let currentFile = null;
let currentFileName = '';
let singleResult = null;
let batchResults = [];
let batchFiles = [];
let modelsData = [];

// ========================
//  Tab switching
// ========================
$$('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    $('#tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ========================
//  Init: load models
// ========================
fetch('/api/models').then(r => r.json()).then(data => {
  modelsData = data.models;
  [('#model-select'), ('#batch-model-select')].forEach(selId => {
    const sel = document.querySelector(selId);
    data.models.forEach((m, i) => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = `${m.name} (${m.size_mb} MB)`;
      if (m.name.includes('phase2')) opt.selected = true;
      sel.appendChild(opt);
    });
  });

  // Model meta
  function updateMeta(selId, metaId) {
    const sel = document.querySelector(selId);
    const meta = document.querySelector(metaId);
    sel.addEventListener('change', () => {
      const m = data.models.find(x => x.name === sel.value);
      meta.textContent = m ? `Backbone: ${m.backbone} · ${m.size_mb} MB` : '';
    });
    sel.dispatchEvent(new Event('change'));
  }
  updateMeta('#model-select', '#model-meta');
  updateMeta('#batch-model-select', '#batch-model-meta');

  // Status bar
  const sb = $('#status-bar');
  const deviceChip = data.device === 'cuda'
    ? `<span class="chip gpu">🟢 GPU (CUDA)</span>`
    : `<span class="chip cpu">🟡 CPU</span>`;
  sb.innerHTML = `${deviceChip} <span class="chip">${data.models.length} models available</span>`;
});

// ========================
//  SINGLE IMAGE
// ========================
const dz = $('#dropzone');
const fi = $('#file-input');
const pw = $('#preview-wrap');
const pi = $('#preview-img');
const pb = $('#predict-btn');
const fnLabel = $('#filename-label');
const fnText = $('#filename-text');

dz.addEventListener('click', () => fi.click());
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fi.addEventListener('change', () => { if (fi.files.length) handleFile(fi.files[0]); });

$('#clear-btn').addEventListener('click', clearSingle);

function clearSingle() {
  currentFile = null;
  currentFileName = '';
  pw.classList.add('hidden');
  fnLabel.classList.add('hidden');
  dz.classList.remove('hidden');
  pb.disabled = true;
  $('#results').classList.remove('visible');
  $('#placeholder').classList.remove('hidden');
  $('#export-single-btn').disabled = true;
  singleResult = null;
}

function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  currentFile = file;
  currentFileName = file.name;
  fnText.textContent = file.name;
  fnLabel.classList.remove('hidden');
  const reader = new FileReader();
  reader.onload = e => {
    pi.src = e.target.result;
    pw.classList.remove('hidden');
    dz.classList.add('hidden');
    pb.disabled = false;
  };
  reader.readAsDataURL(file);
}

// --- Single Predict ---
pb.addEventListener('click', async () => {
  if (!currentFile) return;
  pb.disabled = true;
  pb.classList.add('loading');
  pb.textContent = '⏳ Analyzing...';

  const form = new FormData();
  form.append('image', currentFile);
  form.append('model', $('#model-select').value);
  form.append('filename', currentFileName);

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    singleResult = data;
    showResults(data);
    $('#export-single-btn').disabled = false;
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

  const badge = $('#species-badge');
  badge.textContent = data.species;
  badge.className = 'species-badge ' + data.species.toLowerCase();

  $('#breed-name').textContent = data.top_breed;
  $('#breed-conf').textContent = data.top_breed_confidence.toFixed(1) + '%';

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
    requestAnimationFrame(() => {
      setTimeout(() => {
        li.querySelector('.top5-bar').style.width = barWidth + '%';
      }, i * 100);
    });
  });

  $('#info-footer').innerHTML = `
    <span>🤖 Model: ${data.model_used}</span>
    <span>📄 File: ${data.filename || 'unknown'}</span>
    <span>🐄 Species: ${data.species} (${data.species_confidence.toFixed(1)}%)</span>
  `;
}

// --- Single Export ---
$('#export-single-btn').addEventListener('click', async () => {
  if (!singleResult) return;
  const btn = $('#export-single-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Generating ODT...';

  try {
    const res = await fetch('/api/export-odt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ results: singleResult, mode: 'single' })
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `breed_test_result_${Date.now()}.odt`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert('Export failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '📄 Export to ODT';
  }
});

// ========================
//  BATCH IMAGES
// ========================
const bdz = $('#batch-dropzone');
const bfi = $('#batch-file-input');
const bpb = $('#batch-predict-btn');

bdz.addEventListener('click', () => bfi.click());
bdz.addEventListener('dragover', e => { e.preventDefault(); bdz.classList.add('drag-over'); });
bdz.addEventListener('dragleave', () => bdz.classList.remove('drag-over'));
bdz.addEventListener('drop', e => {
  e.preventDefault();
  bdz.classList.remove('drag-over');
  if (e.dataTransfer.files.length) handleBatchFiles(e.dataTransfer.files);
});
bfi.addEventListener('change', () => { if (bfi.files.length) handleBatchFiles(bfi.files); });

function handleBatchFiles(files) {
  batchFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
  if (!batchFiles.length) return;

  const listEl = $('#batch-file-list');
  listEl.innerHTML = '';
  listEl.classList.remove('hidden');
  bdz.classList.add('hidden');

  batchFiles.forEach(f => {
    const div = document.createElement('div');
    div.className = 'batch-file-item';
    const sizeMB = (f.size / 1024 / 1024).toFixed(2);
    div.innerHTML = `<span class="bf-name">📄 ${f.name}</span><span class="bf-size">${sizeMB} MB</span>`;
    listEl.appendChild(div);
  });

  bpb.disabled = false;
  $('#batch-clear-btn').classList.remove('hidden');
}

// --- Batch Clear ---
$('#batch-clear-btn').addEventListener('click', () => {
  batchFiles = [];
  batchResults = [];
  $('#batch-file-list').classList.add('hidden');
  $('#batch-file-list').innerHTML = '';
  bdz.classList.remove('hidden');
  bpb.disabled = true;
  $('#batch-clear-btn').classList.add('hidden');
  $('#batch-results').classList.add('hidden');
  $('#batch-placeholder').classList.remove('hidden');
  $('#batch-progress-wrap').classList.add('hidden');
  $('#batch-progress-text').classList.add('hidden');
});

// --- Batch Predict ---
bpb.addEventListener('click', async () => {
  if (!batchFiles.length) return;
  bpb.disabled = true;
  bpb.classList.add('loading');
  bpb.textContent = '⏳ Analyzing...';
  batchResults = [];

  const pw = $('#batch-progress-wrap');
  const pbar = $('#batch-progress-bar');
  const ptxt = $('#batch-progress-text');
  pw.classList.remove('hidden');
  ptxt.classList.remove('hidden');

  const model = $('#batch-model-select').value;

  for (let i = 0; i < batchFiles.length; i++) {
    const pct = ((i) / batchFiles.length * 100).toFixed(0);
    pbar.style.width = pct + '%';
    ptxt.textContent = `Processing ${i + 1} / ${batchFiles.length}: ${batchFiles[i].name}`;

    const form = new FormData();
    form.append('image', batchFiles[i]);
    form.append('model', model);
    form.append('filename', batchFiles[i].name);

    try {
      const res = await fetch('/api/predict', { method: 'POST', body: form });
      const data = await res.json();
      data.filename = batchFiles[i].name;
      batchResults.push(data);
    } catch (e) {
      batchResults.push({ error: e.message, filename: batchFiles[i].name });
    }
  }

  pbar.style.width = '100%';
  ptxt.textContent = `Done! ${batchResults.length} images processed.`;

  showBatchResults();

  bpb.disabled = false;
  bpb.classList.remove('loading');
  bpb.textContent = '🔍 Analyze All Images';
});

function showBatchResults() {
  $('#batch-placeholder').classList.add('hidden');
  const wrap = $('#batch-results');
  wrap.classList.remove('hidden');

  const valid = batchResults.filter(r => !r.error);
  const cattle = valid.filter(r => r.species === 'Cattle').length;
  const buffalo = valid.filter(r => r.species === 'Buffalo').length;
  const avgConf = valid.length ? (valid.reduce((s, r) => s + r.top_breed_confidence, 0) / valid.length).toFixed(1) : '0';

  $('#batch-summary').innerHTML = `
    <div><div class="stat-val">${valid.length}</div><div class="stat-label">Images</div></div>
    <div><div class="stat-val">${cattle} / ${buffalo}</div><div class="stat-label">Cattle / Buffalo</div></div>
    <div><div class="stat-val">${avgConf}%</div><div class="stat-label">Avg Confidence</div></div>
  `;

  const list = $('#batch-results-list');
  list.innerHTML = '';

  batchResults.forEach(r => {
    const card = document.createElement('div');
    card.className = 'batch-card';

    if (r.error) {
      card.innerHTML = `
        <div class="bc-header"><span class="bc-filename">📄 ${r.filename}</span></div>
        <div style="color:var(--red)">❌ Error: ${r.error}</div>
      `;
    } else {
      let top5html = '';
      (r.top5_breeds || []).forEach(b => {
        top5html += `<li><span>${b.breed}</span><span>${b.confidence.toFixed(1)}%</span></li>`;
      });
      card.innerHTML = `
        <div class="bc-header">
          <span class="bc-filename">📄 ${r.filename}</span>
          <span class="bc-species ${r.species.toLowerCase()}">${r.species} ${r.species_confidence.toFixed(1)}%</span>
        </div>
        <div class="bc-breed">${r.top_breed}</div>
        <div class="bc-conf">${r.top_breed_confidence.toFixed(1)}% confidence</div>
        <ul class="bc-top5">${top5html}</ul>
      `;
    }
    list.appendChild(card);
  });
}

// --- Batch Export ---
$('#export-batch-btn').addEventListener('click', async () => {
  if (!batchResults.length) return;
  const btn = $('#export-batch-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Generating ODT...';

  try {
    const res = await fetch('/api/export-odt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ results: batchResults, mode: 'batch' })
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `breed_batch_result_${Date.now()}.odt`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert('Export failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '📄 Export All to ODT';
  }
});
</script>
</body>
</html>"""


def run_server(manager, port=8501):
    """Run a minimal HTTP server with the GUI and prediction API."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    html_content = build_html()

    def _parse_multipart(handler):
        """Parse multipart/form-data without the deprecated cgi module."""
        content_type = handler.headers.get("Content-Type", "")
        m = re.search(r'boundary=([^\s;]+)', content_type)
        if not m:
            return {}, {}
        boundary = m.group(1).encode()
        content_length = int(handler.headers.get("Content-Length", 0))
        body = handler.rfile.read(content_length)

        parts_data = {}   # name -> bytes
        parts_text = {}   # name -> str

        chunks = body.split(b"--" + boundary)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or chunk == b"--":
                continue
            sep = chunk.find(b"\r\n\r\n")
            if sep < 0:
                continue
            header_bytes = chunk[:sep]
            part_body = chunk[sep+4:]
            if part_body.endswith(b"\r\n"):
                part_body = part_body[:-2]

            header_str = header_bytes.decode("utf-8", errors="replace")
            name_match = re.search(r'name="([^"]+)"', header_str)
            if not name_match:
                continue
            name = name_match.group(1)

            filename_match = re.search(r'filename="([^"]*)"', header_str)
            if filename_match:
                parts_data[name] = part_body
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
                filename = parts_text.get("filename", "unknown")

                if not image_bytes:
                    self._respond(400, "application/json",
                                  json.dumps({"error": "No image provided"}).encode())
                    return

                result = manager.predict(image_bytes, model_name)
                result["filename"] = filename
                self._respond(200, "application/json", json.dumps(result).encode())

            elif self.path == "/api/export-odt":
                content_length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    results = payload.get("results", {})
                    mode = payload.get("mode", "single")
                    odt_bytes = generate_odt_report(results, mode=mode)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.oasis.opendocument.text")
                    self.send_header("Content-Length", str(len(odt_bytes)))
                    self.send_header("Content-Disposition",
                                     f"attachment; filename=breed_report_{int(time.time())}.odt")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(odt_bytes)
                except Exception as e:
                    self._respond(500, "application/json",
                                  json.dumps({"error": str(e)}).encode())
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
