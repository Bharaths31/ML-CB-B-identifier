#!/usr/bin/env python3
"""
Breed Classifier - Model Tester GUI
========================================

Standalone GUI for testing exported models on individual or batch images.
Upload image(s) (PNG/JPG/JPEG), select a model checkpoint, and get
the predicted species + breed with confidence percentages.
Supports exporting results to ODT format.

Modes:
    python test_model.py                # Developer mode (default)
    python test_model.py --dev          # Developer mode (extra tools & data)
    python test_model.py --present      # Presenter mode (clean, polished view)

Options:
    python test_model.py --port 8501    # Custom port
    python test_model.py --no-browser   # Don't auto-open browser

Opens a browser window with the GUI at http://localhost:8501
"""

import argparse
import base64
import datetime
import glob
import io
import json
import logging
import os
import re
import struct
import sys
import threading
import time
import webbrowser

import torch
import torch.nn.functional as F
from PIL import Image
from PIL.ExifTags import TAGS as EXIF_TAGS
from torchvision import transforms

# ---------------------------------------------------------------------------
#  Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (CHECKPOINT_DIR, EXPORT_DIR, IMAGE_SIZE, LOGS_DIR,
                        PORTABLE_EXPORT_DIR, SPLIT_DIR)
from src.model import BreedClassifier

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

SPECIES_LABELS = {0: "Cattle", 1: "Buffalo"}
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PRESENTER_CONFIG_PATH = os.path.join(LOGS_DIR, "presenter_config.json")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
#  Session Logger
# ---------------------------------------------------------------------------

class SessionLogger:
    """File + stdout logger for the entire GUI session."""

    def __init__(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(LOGS_DIR, f"session_{ts}.log")
        self.start_time = time.time()

        self._logger = logging.getLogger("model_tester")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
                                datefmt="%H:%M:%S")

        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        self._logger.addHandler(sh)

    def info(self, msg):
        self._logger.info(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def error(self, msg):
        self._logger.error(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def stop(self):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        self.info(f"Application stopped (session duration: {mins}m {secs}s)")
        for h in self._logger.handlers[:]:
            h.close()
            self._logger.removeHandler(h)

    def get_log_contents(self):
        """Read the current log file contents."""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""


# Global logger â€” set in main()
logger: SessionLogger = None  # type: ignore


# ---------------------------------------------------------------------------
#  Image metadata extraction
# ---------------------------------------------------------------------------

def extract_image_metadata(image_bytes):
    """Extract detailed metadata from raw image bytes."""
    meta = {}
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Basic dimensions
        meta["width"] = img.width
        meta["height"] = img.height
        meta["aspect_ratio"] = f"{img.width}:{img.height}"
        from math import gcd
        g = gcd(img.width, img.height)
        meta["aspect_ratio_simplified"] = f"{img.width // g}:{img.height // g}"
        meta["megapixels"] = round((img.width * img.height) / 1_000_000, 2)
        meta["orientation"] = "Landscape" if img.width > img.height else (
            "Portrait" if img.height > img.width else "Square")

        # Format & color
        meta["format"] = img.format or "Unknown"
        meta["mode"] = img.mode
        mode_map = {"L": "Grayscale", "LA": "Grayscale+Alpha", "RGB": "RGB",
                    "RGBA": "RGB+Alpha", "CMYK": "CMYK", "P": "Palette",
                    "1": "Binary", "I": "32-bit Integer", "F": "32-bit Float"}
        meta["color_space"] = mode_map.get(img.mode, img.mode)
        meta["channels"] = len(img.getbands())
        meta["bands"] = list(img.getbands())

        if img.mode in ("I", "I;16"):
            meta["bit_depth"] = 16
        elif img.mode == "F":
            meta["bit_depth"] = 32
        elif img.mode == "1":
            meta["bit_depth"] = 1
        else:
            meta["bit_depth"] = 8

        # File size
        meta["file_size_bytes"] = len(image_bytes)
        meta["file_size_kb"] = round(len(image_bytes) / 1024, 2)
        meta["file_size_mb"] = round(len(image_bytes) / (1024 * 1024), 3)

        # DPI
        dpi = img.info.get("dpi")
        if dpi:
            meta["dpi_x"] = round(dpi[0], 1)
            meta["dpi_y"] = round(dpi[1], 1)

        # EXIF data
        exif_data = {}
        try:
            raw_exif = img._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = EXIF_TAGS.get(tag_id, str(tag_id))
                    # Skip binary / large data
                    if isinstance(value, bytes) and len(value) > 100:
                        continue
                    try:
                        json.dumps(value)
                        exif_data[tag_name] = value
                    except (TypeError, ValueError):
                        exif_data[tag_name] = str(value)
        except Exception:
            pass
        meta["exif"] = exif_data

        # Unique colors sample (only for small images)
        if img.width * img.height < 500_000:
            try:
                colors = img.convert("RGB").getcolors(maxcolors=100_000)
                meta["unique_colors"] = len(colors) if colors else "100000+"
            except Exception:
                meta["unique_colors"] = "N/A"
        else:
            meta["unique_colors"] = "N/A (image too large to count)"

    except Exception as e:
        meta["error"] = str(e)

    return meta


# ---------------------------------------------------------------------------
#  Model specification introspection
# ---------------------------------------------------------------------------

def get_model_spec(model, name, info):
    """Extract detailed model architecture and parameter info."""
    spec = {
        "name": name,
        "backbone": info.get("backbone", "unknown"),
        "type": info.get("type", "pt"),
        "path": info.get("path", ""),
        "device": str(next(model.parameters()).device) if hasattr(model, "parameters") else "N/A",
        "image_size": IMAGE_SIZE,
    }

    if info.get("type") == "onnx":
        # ONNX models â€” limited introspection
        try:
            spec["inputs"] = [{"name": i.name, "shape": str(i.shape), "type": i.type}
                              for i in model.get_inputs()]
            spec["outputs"] = [{"name": o.name, "shape": str(o.shape), "type": o.type}
                               for o in model.get_outputs()]
        except Exception:
            pass
        return spec

    # PyTorch models â€” full introspection
    total_params = 0
    trainable_params = 0
    layer_summary = []

    for pname, param in model.named_parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count

    spec["total_parameters"] = total_params
    spec["trainable_parameters"] = trainable_params
    spec["frozen_parameters"] = total_params - trainable_params
    spec["total_parameters_human"] = _human_number(total_params)
    spec["model_size_mb"] = round(total_params * 4 / (1024 * 1024), 2)  # float32

    # Per-component breakdown
    components = {}
    for pname, param in model.named_parameters():
        top_level = pname.split(".")[0]
        if top_level not in components:
            components[top_level] = {"params": 0, "trainable": 0}
        components[top_level]["params"] += param.numel()
        if param.requires_grad:
            components[top_level]["trainable"] += param.numel()

    spec["components"] = {k: {"params": v["params"],
                               "params_human": _human_number(v["params"]),
                               "trainable": v["trainable"]}
                          for k, v in components.items()}

    # Architecture info
    spec["architecture"] = {
        "feature_dim": 1280,
        "binary_head": "Linear(1280â†’256â†’2)",
        "cattle_head": "Linear(1280â†’512â†’57) + Dropout(0.4)",
        "buffalo_head": "Linear(1280â†’512â†’18) + Dropout(0.4)",
        "attention": "CBAM (after stage 3)",
        "pooling": "AdaptiveAvgPool2d(1)",
    }

    # Model info JSON if available
    model_info_path = os.path.join(os.path.dirname(info.get("path", "")), "model_info.json")
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path) as f:
                spec["model_info_json"] = json.load(f)
        except Exception:
            pass

    return spec


def _human_number(n):
    """Format large numbers: 6234567 â†’ '6.23M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
#  Presenter config persistence
# ---------------------------------------------------------------------------

DEFAULT_PRESENTER_CONFIG = {
    "hide_species_confidence": False,
    "hide_top5_list": False,
    "hide_info_footer": False,
    "hide_batch_summary_stats": False,
    "min_breed_confidence_pct": 5.0,
    "min_species_confidence_pct": 70.0,
}


def load_presenter_config():
    """Load presenter config from JSON, creating defaults if missing."""
    try:
        with open(PRESENTER_CONFIG_PATH, "r") as f:
            cfg = json.load(f)
            # Merge with defaults for any missing keys
            merged = {**DEFAULT_PRESENTER_CONFIG, **cfg}
            return merged
    except (FileNotFoundError, json.JSONDecodeError):
        save_presenter_config(DEFAULT_PRESENTER_CONFIG)
        return dict(DEFAULT_PRESENTER_CONFIG)


def save_presenter_config(config):
    """Save presenter config to JSON."""
    with open(PRESENTER_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


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

        if logger:
            for sp, cm in self.class_maps.items():
                logger.info(f"Loaded {sp} class map ({len(cm)} breeds)")

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

        if logger:
            names = ", ".join(found.keys()) if found else "(none)"
            logger.info(f"Found {len(found)} models: {names}")

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
            if logger:
                logger.debug(f"Model '{name}' already cached in memory")
            return info["model"]

        load_start = time.time()
        if logger:
            logger.info(f"Loading model '{name}' from {info['path']}")

        if info.get("type") == "onnx":
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                session = ort.InferenceSession(info["path"], providers=providers)
                info["model"] = session
                if logger:
                    logger.info(f"ONNX model loaded in {time.time() - load_start:.3f}s")
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

        if logger:
            elapsed = time.time() - load_start
            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model loaded in {elapsed:.3f}s ({_human_number(n_params)} params, backbone={backbone})")

        return model

    @torch.no_grad()
    def predict(self, image_bytes, model_name):
        """Run prediction on raw image bytes. Returns result dict."""
        predict_start = time.time()

        if model_name not in self.models:
            if logger:
                logger.error(f"Model '{model_name}' not found")
            return {"error": f"Model '{model_name}' not found"}

        if logger:
            logger.info(f"Image received ({len(image_bytes) / 1024:.1f} KB), model='{model_name}'")

        # Load and preprocess image
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            if logger:
                logger.error(f"Invalid image: {e}")
            return {"error": f"Invalid image: {e}"}

        if logger:
            logger.debug(f"Image decoded: {img.width}Ã—{img.height} â†’ resize to {IMAGE_SIZE}Ã—{IMAGE_SIZE}")

        tensor = TRANSFORM(img).unsqueeze(0)
        info = self.models[model_name]

        try:
            model = self._load_model(model_name)
        except Exception as e:
            if logger:
                logger.error(f"Model load failed: {e}")
            return {"error": str(e)}

        if logger:
            logger.debug(f"Starting inference on tensor shape {list(tensor.shape)}")

        inference_start = time.time()

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

        inference_time = time.time() - inference_start

        # Species prediction
        binary_probs = F.softmax(binary_logits, dim=0)
        species_idx = binary_probs.argmax().item()
        species_name = SPECIES_LABELS[species_idx]
        species_conf = binary_probs[species_idx].item() * 100

        if logger:
            bp = [round(p, 4) for p in binary_probs.tolist()]
            logger.debug(f"Binary softmax: {bp} â†’ {species_name} ({species_conf:.1f}%)")

        # Breed prediction based on species
        if species_idx == 0:  # Cattle
            breed_probs = F.softmax(cattle_logits, dim=0)
            class_map = self.class_maps.get("cattle", {})
            if logger:
                logger.debug("Selected cattle head for breed prediction")
        else:  # Buffalo
            breed_probs = F.softmax(buffalo_logits, dim=0)
            class_map = self.class_maps.get("buffalo", {})
            if logger:
                logger.debug("Selected buffalo head for breed prediction")

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

        total_time = time.time() - predict_start

        if logger:
            top5_str = ", ".join(f"{b['breed']} ({b['confidence']:.1f}%)" for b in top5)
            logger.debug(f"Top-5 breeds: {top5_str}")
            logger.info(f"Prediction: {species_name} â†’ {top5[0]['breed'] if top5 else 'Unknown'} "
                        f"({top5[0]['confidence']:.1f}%) in {total_time:.3f}s "
                        f"(inference: {inference_time:.3f}s)")

        result = {
            "species": species_name,
            "species_confidence": round(species_conf, 2),
            "top_breed": top5[0]["breed"] if top5 else "Unknown",
            "top_breed_confidence": top5[0]["confidence"] if top5 else 0,
            "top5_breeds": top5,
            "model_used": model_name,
            "inference_time_ms": round(inference_time * 1000, 1),
            "total_time_ms": round(total_time * 1000, 1),
        }

        # Add raw logits for dev mode (API consumer decides whether to use them)
        result["_raw_binary_probs"] = [round(p, 6) for p in binary_probs.tolist()]
        result["_raw_breed_probs_top10"] = [
            {"idx": int(idx), "breed": idx_to_breed.get(int(idx), f"class_{int(idx)}"),
             "prob": round(float(p), 6)}
            for p, idx in zip(*breed_probs.topk(min(10, len(breed_probs))))
        ]

        return result


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
    p = P(stylename=title_style, text="ðŸ„ Cattle & Buffalo Breed Classifier â€” Test Report")
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
    p = P(stylename=body_style, text="â”€" * 60)
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
            p = P(stylename=body_style, text="â”€" * 60)
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
                r.get("filename", "â€”"),
                r.get("species", "â€”"),
                r.get("top_breed", "â€”"),
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

def build_html(mode="dev"):
    """Return the complete single-page GUI HTML. Mode: 'dev' or 'present'."""
    is_dev = mode == "dev"

    # Dev tools tab button (only in dev mode)
    dev_tab_btn = '<button class="tab-btn" data-tab="devtools" id="tab-devtools-btn">ðŸ› ï¸ Dev Tools</button>' if is_dev else ''

    # Export buttons visibility
    export_display = '' if is_dev else 'display:none !important;'

    # Mode badge
    mode_badge_style = f'border-color:{"var(--accent)" if is_dev else "var(--green)"}; color:{"var(--accent-light)" if is_dev else "var(--green)"}'
    mode_badge_text = "ðŸ› ï¸ DEV" if is_dev else "ðŸŽ¤ PRESENT"
    mode_badge = f'<span class="chip" style="{mode_badge_style}">{mode_badge_text}</span>'

    # Dev tools tab content (only in dev mode)
    dev_tab_html = ''
    if is_dev:
        dev_tab_html = r"""
<!-- ============================================================ -->
<!--  DEV TOOLS TAB                                                -->
<!-- ============================================================ -->
<div class="tab-content" id="tab-devtools">
<div class="container" style="grid-template-columns:1fr">

  <!-- Model Specification Panel -->
  <div class="card">
    <h2>ðŸ”¬ Model Specification</h2>
    <div id="dev-model-selector" class="model-selector">
      <label for="dev-model-select">Inspect Model</label>
      <select id="dev-model-select"></select>
    </div>
    <button class="predict-btn" id="load-spec-btn" style="margin-bottom:16px">ðŸ“‹ Load Specification</button>
    <div id="model-spec-content" class="dev-panel">
      <div class="dev-placeholder">Select a model and click <strong>Load Specification</strong></div>
    </div>
  </div>

  <!-- Class Maps Viewer -->
  <div class="card">
    <h2>ðŸ“š Class Maps (Breed Data)</h2>
    <div id="class-maps-content" class="dev-panel">
      <div class="dev-placeholder">Loading class maps...</div>
    </div>
  </div>

  <!-- Image Metadata Panel -->
  <div class="card">
    <h2>ðŸ–¼ï¸ Image Metadata Analyzer</h2>
    <div class="dropzone" id="meta-dropzone">
      <span class="icon">ðŸ“</span>
      <p>Drop an image to inspect metadata<br>EXIF, dimensions, color space, and more</p>
      <div class="formats">Supports: PNG, JPG, JPEG, BMP, WebP</div>
      <input type="file" id="meta-file-input" accept=".png,.jpg,.jpeg,.bmp,.webp" hidden>
    </div>
    <div id="image-meta-content" class="dev-panel hidden"></div>
  </div>

  <!-- Presenter Config -->
  <div class="card">
    <h2>ðŸŽ›ï¸ Presenter Config</h2>
    <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:16px">
      Configure what the <code>--present</code> mode shows. Changes persist across restarts.
    </p>
    <div id="presenter-config-panel" class="dev-panel">
      <div class="config-row">
        <label class="toggle-label">
          <input type="checkbox" id="cfg-hide-species-conf">
          <span>Hide species confidence percentage</span>
        </label>
      </div>
      <div class="config-row">
        <label class="toggle-label">
          <input type="checkbox" id="cfg-hide-top5">
          <span>Hide top-5 predictions list</span>
        </label>
      </div>
      <div class="config-row">
        <label class="toggle-label">
          <input type="checkbox" id="cfg-hide-footer">
          <span>Hide info footer</span>
        </label>
      </div>
      <div class="config-row">
        <label class="toggle-label">
          <input type="checkbox" id="cfg-hide-batch-stats">
          <span>Hide batch summary statistics</span>
        </label>
      </div>
      <div class="config-row">
        <label class="range-label">
          <span>Min breed confidence to show: <strong id="cfg-min-breed-val">5%</strong></span>
          <input type="range" id="cfg-min-breed" min="0" max="50" step="1" value="5">
        </label>
      </div>
      <div class="config-row">
        <label class="range-label">
          <span>Min species confidence (show "Low" below): <strong id="cfg-min-species-val">70%</strong></span>
          <input type="range" id="cfg-min-species" min="0" max="95" step="5" value="70">
        </label>
      </div>
      <button class="predict-btn" id="save-presenter-cfg-btn" style="margin-top:12px">ðŸ’¾ Save Config</button>
      <div id="cfg-save-status" style="text-align:center;margin-top:8px;font-size:0.82rem;color:var(--green)"></div>
    </div>
  </div>

  <!-- Session Logs -->
  <div class="card">
    <h2>ðŸ“œ Session Log</h2>
    <button class="predict-btn" id="refresh-logs-btn" style="margin-bottom:12px;background:var(--surface2);color:var(--text-muted);border:1px solid var(--border)">ðŸ”„ Refresh Logs</button>
    <div id="session-log-content" class="dev-panel" style="max-height:400px;overflow-y:auto;">
      <pre style="font-size:0.78rem;color:var(--text-muted);white-space:pre-wrap;word-break:break-all;margin:0">Loading...</pre>
    </div>
  </div>

</div>
</div>
"""

    # Build the JavaScript â€” use .replace to avoid f-string brace conflicts
    js_mode = mode
    js_block = _build_js(mode)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ðŸ„ Breed Classifier â€” Model Tester</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0a0e17; --surface: #111827; --surface2: #1e293b;
      --border: #2d3a4f; --text: #e2e8f0; --text-muted: #94a3b8;
      --accent: #6366f1; --accent-light: #818cf8;
      --green: #22c55e; --green-bg: rgba(34,197,94,0.1);
      --amber: #f59e0b; --amber-bg: rgba(245,158,11,0.1);
      --red: #ef4444; --radius: 16px; --radius-sm: 10px;
    }}
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text);
      min-height: 100vh; overflow-x: hidden;
    }}
    body::before {{
      content: ''; position: fixed; top: -50%; left: -50%;
      width: 200%; height: 200%;
      background: radial-gradient(ellipse at 30% 20%, rgba(99,102,241,0.08) 0%, transparent 50%),
                  radial-gradient(ellipse at 80% 80%, rgba(34,197,94,0.06) 0%, transparent 50%);
      animation: drift 20s ease-in-out infinite; z-index: -1;
    }}
    @keyframes drift {{ 0%,100% {{ transform: translate(0,0); }} 50% {{ transform: translate(-3%,3%); }} }}
    .header {{ text-align: center; padding: 48px 24px 16px; }}
    .header h1 {{
      font-size: 2.2rem; font-weight: 800;
      background: linear-gradient(135deg, var(--accent-light), var(--green));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text; margin-bottom: 6px;
    }}
    .header p {{ color: var(--text-muted); font-size: 0.95rem; font-weight: 300; }}
    .tabs {{
      display: flex; justify-content: center; gap: 4px;
      margin: 16px auto 24px; background: var(--surface);
      border-radius: 12px; padding: 4px; width: fit-content;
      border: 1px solid var(--border);
    }}
    .tab-btn {{
      padding: 10px 28px; border: none; border-radius: 9px;
      background: transparent; color: var(--text-muted);
      font-family: inherit; font-size: 0.9rem; font-weight: 500;
      cursor: pointer; transition: all 0.25s;
    }}
    .tab-btn:hover {{ color: var(--text); background: var(--surface2); }}
    .tab-btn.active {{
      background: linear-gradient(135deg, var(--accent), #8b5cf6);
      color: white; font-weight: 600;
    }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    .container {{
      max-width: 1100px; margin: 0 auto; padding: 0 24px 60px;
      display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
    }}
    @media (max-width: 768px) {{ .container {{ grid-template-columns: 1fr; }} }}
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 28px; transition: border-color 0.3s;
    }}
    .card:hover {{ border-color: var(--accent); }}
    .card h2 {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 18px; display: flex; align-items: center; gap: 8px; }}
    .model-selector {{ background: var(--surface2); border-radius: var(--radius-sm); padding: 14px 16px; margin-bottom: 20px; }}
    .model-selector label {{ display: block; font-size: 0.8rem; font-weight: 500; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }}
    .model-selector select {{ width: 100%; background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 10px 14px; border-radius: 8px; font-size: 0.9rem; font-family: inherit; cursor: pointer; outline: none; transition: border-color 0.2s; }}
    .model-selector select:focus {{ border-color: var(--accent); }}
    .model-meta {{ margin-top: 8px; font-size: 0.78rem; color: var(--text-muted); }}
    .dropzone {{ border: 2px dashed var(--border); border-radius: var(--radius-sm); padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.3s; position: relative; }}
    .dropzone:hover, .dropzone.drag-over {{ border-color: var(--accent); background: rgba(99,102,241,0.05); }}
    .dropzone .icon {{ font-size: 3rem; margin-bottom: 12px; display: block; }}
    .dropzone p {{ color: var(--text-muted); font-size: 0.88rem; line-height: 1.6; }}
    .dropzone .formats {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 6px; opacity: 0.7; }}
    .filename-label {{ background: var(--surface2); border-radius: 8px; padding: 8px 14px; margin-bottom: 12px; font-size: 0.82rem; color: var(--text-muted); display: flex; align-items: center; gap: 6px; word-break: break-all; }}
    .filename-label .fname {{ color: var(--text); font-weight: 500; }}
    .preview-container {{ position: relative; border-radius: var(--radius-sm); overflow: hidden; margin-bottom: 16px; }}
    .preview-container img {{ width: 100%; height: 260px; object-fit: cover; display: block; border-radius: var(--radius-sm); }}
    .clear-btn {{ position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.6); backdrop-filter: blur(8px); border: none; color: white; width: 32px; height: 32px; border-radius: 50%; cursor: pointer; font-size: 1rem; display: flex; align-items: center; justify-content: center; transition: background 0.2s; }}
    .clear-btn:hover {{ background: var(--red); }}
    .predict-btn {{ width: 100%; padding: 14px; border: none; border-radius: var(--radius-sm); background: linear-gradient(135deg, var(--accent), #8b5cf6); color: white; font-size: 1rem; font-weight: 600; font-family: inherit; cursor: pointer; transition: all 0.3s; position: relative; overflow: hidden; }}
    .predict-btn:hover:not(:disabled) {{ transform: translateY(-1px); box-shadow: 0 8px 25px rgba(99,102,241,0.3); }}
    .predict-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    .predict-btn.loading {{ pointer-events: none; }}
    .predict-btn.loading::after {{ content: ''; position: absolute; top: 0; left: -100%; width: 200%; height: 100%; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent); animation: shimmer 1.5s infinite; }}
    @keyframes shimmer {{ 100% {{ transform: translateX(100%); }} }}
    .export-btn {{ width: 100%; padding: 12px; border: 1px solid var(--green); border-radius: var(--radius-sm); background: transparent; color: var(--green); font-size: 0.9rem; font-weight: 600; font-family: inherit; cursor: pointer; transition: all 0.3s; margin-top: 12px; {export_display} }}
    .export-btn:hover:not(:disabled) {{ background: var(--green-bg); transform: translateY(-1px); }}
    .export-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    .results {{ display: none; }}
    .results.visible {{ display: block; animation: fadeUp 0.5s ease; }}
    @keyframes fadeUp {{ from {{ opacity:0; transform:translateY(12px); }} to {{ opacity:1; transform:translateY(0); }} }}
    .result-hero {{ background: var(--surface2); border-radius: var(--radius-sm); padding: 24px; text-align: center; margin-bottom: 16px; border: 1px solid var(--border); }}
    .result-hero .species-badge {{ display: inline-block; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px; }}
    .result-hero .species-badge.cattle {{ background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(245,158,11,0.3); }}
    .result-hero .species-badge.buffalo {{ background: var(--green-bg); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }}
    .result-hero .breed-name {{ font-size: 1.8rem; font-weight: 800; margin-bottom: 4px; line-height: 1.2; }}
    .result-hero .confidence {{ font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, var(--green), #4ade80); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }}
    .result-hero .conf-label {{ font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }}
    .top5-list {{ list-style: none; }}
    .top5-item {{ display: grid; grid-template-columns: 1fr 60px; align-items: center; gap: 12px; margin-bottom: 10px; }}
    .top5-bar-wrap {{ position: relative; height: 32px; background: var(--surface2); border-radius: 6px; overflow: hidden; }}
    .top5-bar {{ height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--accent), var(--accent-light)); transition: width 0.8s cubic-bezier(0.22,1,0.36,1); }}
    .top5-bar-wrap .breed-label {{ position: absolute; top: 50%; left: 12px; transform: translateY(-50%); font-size: 0.82rem; font-weight: 500; color: white; white-space: nowrap; text-shadow: 0 1px 3px rgba(0,0,0,0.5); }}
    .top5-pct {{ font-size: 0.85rem; font-weight: 600; color: var(--text-muted); text-align: right; }}
    .info-footer {{ margin-top: 16px; padding: 12px 16px; background: var(--surface2); border-radius: 8px; font-size: 0.78rem; color: var(--text-muted); display: flex; gap: 16px; flex-wrap: wrap; }}
    .info-footer span {{ display: flex; align-items: center; gap: 4px; }}
    .status-bar {{ text-align: center; padding: 8px; font-size: 0.78rem; color: var(--text-muted); }}
    .status-bar .chip {{ display: inline-block; padding: 4px 12px; border-radius: 12px; background: var(--surface); border: 1px solid var(--border); margin: 0 4px; }}
    .chip.gpu {{ border-color: var(--green); color: var(--green); }}
    .chip.cpu {{ border-color: var(--amber); color: var(--amber); }}
    .batch-results-wrap {{ max-height: 70vh; overflow-y: auto; }}
    .batch-card {{ background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 18px; margin-bottom: 14px; transition: border-color 0.2s; }}
    .batch-card:hover {{ border-color: var(--accent); }}
    .batch-card .bc-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .batch-card .bc-filename {{ font-size: 0.85rem; font-weight: 600; color: var(--accent-light); word-break: break-all; }}
    .batch-card .bc-species {{ padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; }}
    .batch-card .bc-species.cattle {{ background: var(--amber-bg); color: var(--amber); }}
    .batch-card .bc-species.buffalo {{ background: var(--green-bg); color: var(--green); }}
    .batch-card .bc-breed {{ font-size: 1.15rem; font-weight: 700; margin-bottom: 4px; }}
    .batch-card .bc-conf {{ font-size: 0.85rem; color: var(--green); font-weight: 600; }}
    .batch-card .bc-top5 {{ margin-top: 10px; list-style: none; font-size: 0.8rem; }}
    .batch-card .bc-top5 li {{ display: flex; justify-content: space-between; padding: 3px 0; color: var(--text-muted); border-bottom: 1px solid rgba(255,255,255,0.04); }}
    .batch-card .bc-top5 li:last-child {{ border-bottom: none; }}
    .batch-file-list {{ max-height: 200px; overflow-y: auto; background: var(--surface2); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; }}
    .batch-file-item {{ display: flex; justify-content: space-between; align-items: center; padding: 5px 0; font-size: 0.82rem; border-bottom: 1px solid rgba(255,255,255,0.04); }}
    .batch-file-item:last-child {{ border-bottom: none; }}
    .batch-file-item .bf-name {{ color: var(--text); word-break: break-all; }}
    .batch-file-item .bf-size {{ color: var(--text-muted); white-space: nowrap; margin-left: 12px; }}
    .progress-wrap {{ margin: 12px 0; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }}
    .progress-bar {{ height: 100%; background: linear-gradient(90deg, var(--accent), var(--green)); border-radius: 3px; transition: width 0.3s; }}
    .progress-text {{ text-align: center; font-size: 0.78rem; color: var(--text-muted); margin-top: 4px; }}
    .batch-summary {{ background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 18px; margin-bottom: 14px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; text-align: center; }}
    .batch-summary .stat-val {{ font-size: 1.6rem; font-weight: 800; background: linear-gradient(135deg, var(--accent-light), var(--green)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }}
    .batch-summary .stat-label {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }}
    .hidden {{ display: none !important; }}
    .dev-panel {{ background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 18px; margin-top: 8px; }}
    .dev-placeholder {{ text-align: center; padding: 30px; color: var(--text-muted); }}
    .dev-section {{ margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
    .dev-section:last-child {{ border-bottom: none; margin-bottom: 0; padding-bottom: 0; }}
    .dev-section h3 {{ font-size: 0.9rem; font-weight: 600; color: var(--accent-light); margin-bottom: 10px; display: flex; align-items: center; gap: 6px; }}
    .dev-kv {{ display: grid; grid-template-columns: 180px 1fr; gap: 6px 16px; font-size: 0.84rem; }}
    .dev-kv .k {{ color: var(--text-muted); font-weight: 500; }}
    .dev-kv .v {{ color: var(--text); word-break: break-all; }}
    .dev-json {{ background: var(--surface2); border-radius: 8px; padding: 14px; font-family: 'SF Mono', Monaco, Consolas, monospace; font-size: 0.78rem; color: var(--green); max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }}
    .collapsible-header {{ cursor: pointer; padding: 10px 14px; background: var(--surface2); border-radius: 8px; font-size: 0.85rem; font-weight: 600; color: var(--accent-light); display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; transition: background 0.2s; }}
    .collapsible-header:hover {{ background: var(--border); }}
    .collapsible-body {{ display: none; }}
    .collapsible-body.open {{ display: block; }}
    .config-row {{ padding: 10px 0; border-bottom: 1px solid var(--border); }}
    .config-row:last-child {{ border-bottom: none; }}
    .toggle-label, .range-label {{ display: flex; align-items: center; gap: 12px; font-size: 0.88rem; color: var(--text); cursor: pointer; }}
    .range-label {{ flex-direction: column; align-items: flex-start; gap: 8px; }}
    .range-label input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    .toggle-label input[type="checkbox"] {{ width: 18px; height: 18px; accent-color: var(--accent); cursor: pointer; }}
    .dev-comp-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 8px; }}
    .dev-comp-table th {{ text-align: left; padding: 8px 12px; background: var(--surface2); color: var(--text-muted); font-weight: 600; border-bottom: 1px solid var(--border); }}
    .dev-comp-table td {{ padding: 6px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); color: var(--text); }}
    .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    @media (max-width: 768px) {{ .meta-grid {{ grid-template-columns: 1fr; }} }}
    .meta-card {{ background: var(--surface2); border-radius: 8px; padding: 14px; border: 1px solid var(--border); }}
    .meta-card h4 {{ font-size: 0.82rem; color: var(--accent-light); margin-bottom: 8px; }}
    .timing-badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 600; margin-left: 8px; background: rgba(99,102,241,0.15); color: var(--accent-light); border: 1px solid rgba(99,102,241,0.3); }}
  </style>
</head>
<body>
<div class="header">
  <h1>ðŸ„ Breed Classifier â€” Model Tester</h1>
  <p>Upload images to identify cattle and buffalo breeds with confidence scores</p>
</div>
<div class="status-bar" id="status-bar">{mode_badge}</div>
<div class="tabs">
  <button class="tab-btn active" data-tab="single" id="tab-single-btn">ðŸ“¸ Single Image</button>
  <button class="tab-btn" data-tab="batch" id="tab-batch-btn">ðŸ“ Batch Images</button>
  {dev_tab_btn}
</div>

<!-- SINGLE IMAGE TAB -->
<div class="tab-content active" id="tab-single">
<div class="container">
  <div>
    <div class="card">
      <h2>ðŸ“¸ Image Upload</h2>
      <div class="dropzone" id="dropzone">
        <span class="icon">ðŸ“·</span>
        <p>Drag & drop an image here<br>or click to browse</p>
        <div class="formats">Supports: PNG, JPG, JPEG, BMP, WebP</div>
        <input type="file" id="file-input" accept=".png,.jpg,.jpeg,.bmp,.webp" hidden>
      </div>
      <div class="filename-label hidden" id="filename-label">ðŸ“„ <span class="fname" id="filename-text"></span></div>
      <div class="preview-container hidden" id="preview-wrap">
        <img id="preview-img" alt="Preview">
        <button class="clear-btn" id="clear-btn" title="Clear image">âœ•</button>
      </div>
      <div class="model-selector">
        <label for="model-select">Select Model</label>
        <select id="model-select"></select>
        <div class="model-meta" id="model-meta"></div>
      </div>
      <button class="predict-btn" id="predict-btn" disabled>ðŸ” Analyze Breed</button>
    </div>
  </div>
  <div>
    <div class="card">
      <h2>ðŸ“Š Prediction Results</h2>
      <div class="results" id="results">
        <div class="result-hero" id="result-hero">
          <div class="species-badge" id="species-badge"></div>
          <div class="breed-name" id="breed-name"></div>
          <div class="confidence" id="breed-conf"></div>
          <div class="conf-label">Breed Confidence</div>
        </div>
        <h2 style="margin-top:20px" id="top5-heading">ðŸ† Top 5 Predictions</h2>
        <ul class="top5-list" id="top5-list"></ul>
        <div class="info-footer" id="info-footer"></div>
        <button class="export-btn" id="export-single-btn" disabled>ðŸ“„ Export to ODT</button>
      </div>
      <div id="placeholder" style="text-align:center;padding:60px 20px;color:var(--text-muted)">
        <span style="font-size:3rem;display:block;margin-bottom:12px">ðŸ”¬</span>
        <p>Upload an image and click <strong>Analyze Breed</strong><br>to see predictions here</p>
      </div>
    </div>
  </div>
</div>
</div>

<!-- BATCH IMAGE TAB -->
<div class="tab-content" id="tab-batch">
<div class="container">
  <div>
    <div class="card">
      <h2>ðŸ“ Batch Upload</h2>
      <div class="dropzone" id="batch-dropzone">
        <span class="icon">ðŸ“‚</span>
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
      <button class="predict-btn" id="batch-predict-btn" disabled>ðŸ” Analyze All Images</button>
      <div class="progress-wrap hidden" id="batch-progress-wrap"><div class="progress-bar" id="batch-progress-bar" style="width:0%"></div></div>
      <div class="progress-text hidden" id="batch-progress-text"></div>
      <button class="predict-btn hidden" id="batch-clear-btn" style="background:var(--surface2);color:var(--text-muted);margin-top:10px;border:1px solid var(--border)">ðŸ—‘ï¸ Clear All</button>
    </div>
  </div>
  <div>
    <div class="card">
      <h2>ðŸ“Š Batch Results</h2>
      <div class="hidden" id="batch-results">
        <div class="batch-summary" id="batch-summary"></div>
        <button class="export-btn" id="export-batch-btn">ðŸ“„ Export All to ODT</button>
        <div class="batch-results-wrap" id="batch-results-list" style="margin-top:14px"></div>
      </div>
      <div id="batch-placeholder" style="text-align:center;padding:60px 20px;color:var(--text-muted)">
        <span style="font-size:3rem;display:block;margin-bottom:12px">ðŸ“Š</span>
        <p>Select images and click <strong>Analyze All</strong><br>to see batch results here</p>
      </div>
    </div>
  </div>
</div>
</div>

{dev_tab_html}

<script>
{js_block}
</script>
</body>
</html>'''


def _build_js(mode):
    """Build the JavaScript block â€” plain string, no f-string escaping issues."""
    return """
const MODE = '""" + mode + """';
const IS_DEV = MODE === 'dev';
const IS_PRESENT = MODE === 'present';
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
let currentFile = null, currentFileName = '', singleResult = null;
let batchResults = [], batchFiles = [], modelsData = [], presenterConfig = {};

// Tab switching
$$('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    $('#tab-' + btn.dataset.tab).classList.add('active');
  });
});

// Init: load models
fetch('/api/models').then(r => r.json()).then(data => {
  modelsData = data.models;
  const sels = ['#model-select', '#batch-model-select'];
  if (IS_DEV) sels.push('#dev-model-select');
  sels.forEach(selId => {
    const sel = document.querySelector(selId);
    if (!sel) return;
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.name + ' (' + m.size_mb + ' MB)';
      if (m.name.includes('phase2')) opt.selected = true;
      sel.appendChild(opt);
    });
  });
  function updateMeta(selId, metaId) {
    const sel = document.querySelector(selId);
    const meta = document.querySelector(metaId);
    if (!sel || !meta) return;
    sel.addEventListener('change', () => {
      const m = data.models.find(x => x.name === sel.value);
      meta.textContent = m ? 'Backbone: ' + m.backbone + ' \\u00b7 ' + m.size_mb + ' MB' : '';
    });
    sel.dispatchEvent(new Event('change'));
  }
  updateMeta('#model-select', '#model-meta');
  updateMeta('#batch-model-select', '#batch-model-meta');
  const sb = $('#status-bar');
  const dc = data.device === 'cuda'
    ? '<span class="chip gpu">\\ud83d\\udfe2 GPU (CUDA)</span>'
    : '<span class="chip cpu">\\ud83d\\udfe1 CPU</span>';
  sb.innerHTML = dc + ' <span class="chip">' + data.models.length + ' models</span> ' + sb.innerHTML;
});

// Load presenter config
fetch('/api/presenter-config').then(r => r.json()).then(cfg => {
  presenterConfig = cfg;
  if (IS_DEV) {
    const el = id => document.getElementById(id);
    if (el('cfg-hide-species-conf')) el('cfg-hide-species-conf').checked = cfg.hide_species_confidence || false;
    if (el('cfg-hide-top5')) el('cfg-hide-top5').checked = cfg.hide_top5_list || false;
    if (el('cfg-hide-footer')) el('cfg-hide-footer').checked = cfg.hide_info_footer || false;
    if (el('cfg-hide-batch-stats')) el('cfg-hide-batch-stats').checked = cfg.hide_batch_summary_stats || false;
    if (el('cfg-min-breed')) { el('cfg-min-breed').value = cfg.min_breed_confidence_pct || 5; el('cfg-min-breed-val').textContent = (cfg.min_breed_confidence_pct || 5) + '%'; }
    if (el('cfg-min-species')) { el('cfg-min-species').value = cfg.min_species_confidence_pct || 70; el('cfg-min-species-val').textContent = (cfg.min_species_confidence_pct || 70) + '%'; }
  }
}).catch(() => {});

// SINGLE IMAGE
const dz = $('#dropzone'), fi = $('#file-input'), pw = $('#preview-wrap'), pi = $('#preview-img'), pb = $('#predict-btn');
const fnLabel = $('#filename-label'), fnText = $('#filename-text');
dz.addEventListener('click', () => fi.click());
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag-over'); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
fi.addEventListener('change', () => { if (fi.files.length) handleFile(fi.files[0]); });
$('#clear-btn').addEventListener('click', clearSingle);

function clearSingle() {
  currentFile = null; currentFileName = '';
  pw.classList.add('hidden'); fnLabel.classList.add('hidden');
  dz.classList.remove('hidden'); pb.disabled = true;
  $('#results').classList.remove('visible');
  $('#placeholder').classList.remove('hidden');
  const eb = $('#export-single-btn'); if (eb) eb.disabled = true;
  singleResult = null;
}
function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  currentFile = file; currentFileName = file.name;
  fnText.textContent = file.name; fnLabel.classList.remove('hidden');
  const reader = new FileReader();
  reader.onload = e => { pi.src = e.target.result; pw.classList.remove('hidden'); dz.classList.add('hidden'); pb.disabled = false; };
  reader.readAsDataURL(file);
}

pb.addEventListener('click', async () => {
  if (!currentFile) return;
  pb.disabled = true; pb.classList.add('loading'); pb.textContent = '\\u23f3 Analyzing...';
  const form = new FormData();
  form.append('image', currentFile); form.append('model', $('#model-select').value); form.append('filename', currentFileName);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    singleResult = data; showResults(data);
    const eb = $('#export-single-btn'); if (eb) eb.disabled = false;
  } catch (e) { alert('Prediction failed: ' + e.message); }
  finally { pb.disabled = false; pb.classList.remove('loading'); pb.textContent = '\\ud83d\\udd0d Analyze Breed'; }
});

function showResults(data) {
  $('#placeholder').classList.add('hidden');
  $('#results').classList.add('visible');
  const badge = $('#species-badge');
  const minSp = presenterConfig.min_species_confidence_pct || 70;
  if (IS_PRESENT && presenterConfig.hide_species_confidence) {
    badge.textContent = data.species;
  } else if (IS_PRESENT && data.species_confidence < minSp) {
    badge.textContent = data.species + ' (Low)';
  } else {
    badge.textContent = data.species + ' ' + data.species_confidence.toFixed(1) + '%';
  }
  badge.className = 'species-badge ' + data.species.toLowerCase();
  $('#breed-name').textContent = data.top_breed;
  const confEl = $('#breed-conf');
  confEl.textContent = data.top_breed_confidence.toFixed(1) + '%';
  if (IS_DEV && data.inference_time_ms) confEl.innerHTML += '<span class="timing-badge">\\u26a1 ' + data.inference_time_ms + 'ms</span>';

  const list = $('#top5-list'), th = $('#top5-heading');
  list.innerHTML = '';
  if (IS_PRESENT && presenterConfig.hide_top5_list) { th.style.display = 'none'; list.style.display = 'none'; }
  else {
    th.style.display = ''; list.style.display = '';
    const minC = IS_PRESENT ? (presenterConfig.min_breed_confidence_pct || 5) : 0;
    const fb = data.top5_breeds.filter(b => b.confidence >= minC);
    const mx = Math.max(...fb.map(b => b.confidence), 1);
    fb.forEach((b, i) => {
      const li = document.createElement('li'); li.className = 'top5-item';
      const bw = Math.max(8, (b.confidence / mx) * 100);
      li.innerHTML = '<div class="top5-bar-wrap"><div class="top5-bar" style="width:0%"></div><span class="breed-label">' + b.breed + '</span></div><span class="top5-pct">' + b.confidence.toFixed(1) + '%</span>';
      list.appendChild(li);
      requestAnimationFrame(() => { setTimeout(() => { li.querySelector('.top5-bar').style.width = bw + '%'; }, i * 100); });
    });
  }
  const footer = $('#info-footer');
  if (IS_PRESENT && presenterConfig.hide_info_footer) { footer.style.display = 'none'; }
  else {
    footer.style.display = '';
    let fh = '<span>\\ud83e\\udd16 Model: ' + data.model_used + '</span>';
    if (IS_DEV) {
      fh += '<span>\\ud83d\\udcc4 File: ' + (data.filename || 'unknown') + '</span>';
      fh += '<span>\\ud83d\\udc04 Species: ' + data.species + ' (' + data.species_confidence.toFixed(1) + '%)</span>';
      fh += '<span>\\u23f1\\ufe0f ' + (data.total_time_ms || 0) + 'ms total</span>';
    } else { fh += '<span>\\ud83d\\udc04 ' + data.species + '</span>'; }
    footer.innerHTML = fh;
  }
}

// Single Export
const esb = $('#export-single-btn');
if (esb) esb.addEventListener('click', async () => {
  if (!singleResult) return;
  esb.disabled = true; esb.textContent = '\\u23f3 Generating ODT...';
  try {
    const res = await fetch('/api/export-odt', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ results: singleResult, mode: 'single' }) });
    if (!res.ok) { const err = await res.json(); alert(err.error || 'Export failed'); return; }
    const blob = await res.blob(); const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'breed_test_result_' + Date.now() + '.odt'; a.click(); URL.revokeObjectURL(url);
  } catch (e) { alert('Export failed: ' + e.message); }
  finally { esb.disabled = false; esb.textContent = '\\ud83d\\udcc4 Export to ODT'; }
});

// BATCH IMAGES
const bdz = $('#batch-dropzone'), bfi = $('#batch-file-input'), bpb = $('#batch-predict-btn');
bdz.addEventListener('click', () => bfi.click());
bdz.addEventListener('dragover', e => { e.preventDefault(); bdz.classList.add('drag-over'); });
bdz.addEventListener('dragleave', () => bdz.classList.remove('drag-over'));
bdz.addEventListener('drop', e => { e.preventDefault(); bdz.classList.remove('drag-over'); if (e.dataTransfer.files.length) handleBatchFiles(e.dataTransfer.files); });
bfi.addEventListener('change', () => { if (bfi.files.length) handleBatchFiles(bfi.files); });

function handleBatchFiles(files) {
  batchFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
  if (!batchFiles.length) return;
  const listEl = $('#batch-file-list'); listEl.innerHTML = ''; listEl.classList.remove('hidden'); bdz.classList.add('hidden');
  batchFiles.forEach(f => {
    const div = document.createElement('div'); div.className = 'batch-file-item';
    div.innerHTML = '<span class="bf-name">\\ud83d\\udcc4 ' + f.name + '</span><span class="bf-size">' + (f.size/1024/1024).toFixed(2) + ' MB</span>';
    listEl.appendChild(div);
  });
  bpb.disabled = false; $('#batch-clear-btn').classList.remove('hidden');
}

$('#batch-clear-btn').addEventListener('click', () => {
  batchFiles = []; batchResults = [];
  $('#batch-file-list').classList.add('hidden'); $('#batch-file-list').innerHTML = '';
  bdz.classList.remove('hidden'); bpb.disabled = true;
  $('#batch-clear-btn').classList.add('hidden'); $('#batch-results').classList.add('hidden');
  $('#batch-placeholder').classList.remove('hidden');
  $('#batch-progress-wrap').classList.add('hidden'); $('#batch-progress-text').classList.add('hidden');
});

bpb.addEventListener('click', async () => {
  if (!batchFiles.length) return;
  bpb.disabled = true; bpb.classList.add('loading'); bpb.textContent = '\\u23f3 Analyzing...';
  batchResults = [];
  const bpw = $('#batch-progress-wrap'), bpbar = $('#batch-progress-bar'), bptxt = $('#batch-progress-text');
  bpw.classList.remove('hidden'); bptxt.classList.remove('hidden');
  const model = $('#batch-model-select').value;
  for (let i = 0; i < batchFiles.length; i++) {
    bpbar.style.width = ((i) / batchFiles.length * 100).toFixed(0) + '%';
    bptxt.textContent = 'Processing ' + (i+1) + ' / ' + batchFiles.length + ': ' + batchFiles[i].name;
    const form = new FormData();
    form.append('image', batchFiles[i]); form.append('model', model); form.append('filename', batchFiles[i].name);
    try { const res = await fetch('/api/predict', { method: 'POST', body: form }); const data = await res.json(); data.filename = batchFiles[i].name; batchResults.push(data); }
    catch (e) { batchResults.push({ error: e.message, filename: batchFiles[i].name }); }
  }
  bpbar.style.width = '100%'; bptxt.textContent = 'Done! ' + batchResults.length + ' images processed.';
  showBatchResults();
  bpb.disabled = false; bpb.classList.remove('loading'); bpb.textContent = '\\ud83d\\udd0d Analyze All Images';
});

function showBatchResults() {
  $('#batch-placeholder').classList.add('hidden'); $('#batch-results').classList.remove('hidden');
  const valid = batchResults.filter(r => !r.error);
  const cattle = valid.filter(r => r.species === 'Cattle').length;
  const buffalo = valid.filter(r => r.species === 'Buffalo').length;
  const avgConf = valid.length ? (valid.reduce((s, r) => s + r.top_breed_confidence, 0) / valid.length).toFixed(1) : '0';
  const bs = $('#batch-summary');
  if (IS_PRESENT && presenterConfig.hide_batch_summary_stats) { bs.style.display = 'none'; }
  else { bs.style.display = ''; bs.innerHTML = '<div><div class="stat-val">' + valid.length + '</div><div class="stat-label">Images</div></div><div><div class="stat-val">' + cattle + ' / ' + buffalo + '</div><div class="stat-label">Cattle / Buffalo</div></div><div><div class="stat-val">' + avgConf + '%</div><div class="stat-label">Avg Confidence</div></div>'; }
  const list = $('#batch-results-list'); list.innerHTML = '';
  const minC = IS_PRESENT ? (presenterConfig.min_breed_confidence_pct || 5) : 0;
  batchResults.forEach(r => {
    const card = document.createElement('div'); card.className = 'batch-card';
    if (r.error) { card.innerHTML = '<div class="bc-header"><span class="bc-filename">\\ud83d\\udcc4 ' + r.filename + '</span></div><div style="color:var(--red)">\\u274c Error: ' + r.error + '</div>'; }
    else {
      let t5 = ''; (r.top5_breeds || []).filter(b => b.confidence >= minC).forEach(b => { t5 += '<li><span>' + b.breed + '</span><span>' + b.confidence.toFixed(1) + '%</span></li>'; });
      let st = r.species; if (!IS_PRESENT || !presenterConfig.hide_species_confidence) st += ' ' + r.species_confidence.toFixed(1) + '%';
      card.innerHTML = '<div class="bc-header"><span class="bc-filename">\\ud83d\\udcc4 ' + r.filename + '</span><span class="bc-species ' + r.species.toLowerCase() + '">' + st + '</span></div><div class="bc-breed">' + r.top_breed + '</div><div class="bc-conf">' + r.top_breed_confidence.toFixed(1) + '% confidence</div><ul class="bc-top5">' + t5 + '</ul>';
    }
    list.appendChild(card);
  });
}

const ebb = $('#export-batch-btn');
if (ebb) ebb.addEventListener('click', async () => {
  if (!batchResults.length) return;
  ebb.disabled = true; ebb.textContent = '\\u23f3 Generating ODT...';
  try {
    const res = await fetch('/api/export-odt', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ results: batchResults, mode: 'batch' }) });
    if (!res.ok) { const err = await res.json(); alert(err.error || 'Export failed'); return; }
    const blob = await res.blob(); const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'breed_batch_result_' + Date.now() + '.odt'; a.click(); URL.revokeObjectURL(url);
  } catch (e) { alert('Export failed: ' + e.message); }
  finally { ebb.disabled = false; ebb.textContent = '\\ud83d\\udcc4 Export All to ODT'; }
});

// DEV TOOLS
if (IS_DEV) {
  fetch('/api/dev-info').then(r => r.json()).then(info => {
    const el = $('#class-maps-content'); let html = '';
    if (info.cattle_classes) {
      const c = Object.keys(info.cattle_classes).length;
      html += '<div class="collapsible-header" onclick="this.nextElementSibling.classList.toggle(\'open\')">\\ud83d\\udc04 Cattle Breeds (' + c + ') <span>\\u25bc</span></div><div class="collapsible-body"><div class="dev-json">' + JSON.stringify(info.cattle_classes, null, 2) + '</div></div>';
    }
    if (info.buffalo_classes) {
      const c = Object.keys(info.buffalo_classes).length;
      html += '<div class="collapsible-header" onclick="this.nextElementSibling.classList.toggle(\'open\')" style="margin-top:8px">\\ud83d\\udc03 Buffalo Breeds (' + c + ') <span>\\u25bc</span></div><div class="collapsible-body"><div class="dev-json">' + JSON.stringify(info.buffalo_classes, null, 2) + '</div></div>';
    }
    if (!html) html = '<div class="dev-placeholder">No class maps found</div>';
    el.innerHTML = html;
  }).catch(() => {});

  const lsb = $('#load-spec-btn');
  if (lsb) lsb.addEventListener('click', async () => {
    const mn = $('#dev-model-select').value; if (!mn) return;
    lsb.disabled = true; lsb.textContent = '\\u23f3 Loading...';
    try {
      const res = await fetch('/api/model-spec?model=' + encodeURIComponent(mn));
      const spec = await res.json();
      const el = $('#model-spec-content');
      let h = '<div class="dev-section"><h3>\\ud83d\\udccb Overview</h3><div class="dev-kv">';
      h += '<span class="k">Name</span><span class="v">' + (spec.name||'N/A') + '</span>';
      h += '<span class="k">Backbone</span><span class="v">' + (spec.backbone||'N/A') + '</span>';
      h += '<span class="k">Type</span><span class="v">' + (spec.type||'N/A') + '</span>';
      h += '<span class="k">Device</span><span class="v">' + (spec.device||'N/A') + '</span>';
      h += '<span class="k">Input Size</span><span class="v">' + (spec.image_size||260) + '\\u00d7' + (spec.image_size||260) + ' RGB</span>';
      h += '<span class="k">Path</span><span class="v" style="font-size:0.75rem">' + (spec.path||'N/A') + '</span>';
      h += '</div></div>';
      if (spec.total_parameters) {
        h += '<div class="dev-section"><h3>\\ud83d\\udcca Parameters</h3><div class="dev-kv">';
        h += '<span class="k">Total</span><span class="v">' + spec.total_parameters_human + ' (' + spec.total_parameters.toLocaleString() + ')</span>';
        h += '<span class="k">Trainable</span><span class="v">' + spec.trainable_parameters.toLocaleString() + '</span>';
        h += '<span class="k">Frozen</span><span class="v">' + spec.frozen_parameters.toLocaleString() + '</span>';
        h += '<span class="k">Est. Size</span><span class="v">' + spec.model_size_mb + ' MB (float32)</span>';
        h += '</div></div>';
      }
      if (spec.architecture) {
        h += '<div class="dev-section"><h3>\\ud83c\\udfd7\\ufe0f Architecture</h3><div class="dev-kv">';
        for (const [k,v] of Object.entries(spec.architecture)) h += '<span class="k">' + k.replace(/_/g,' ') + '</span><span class="v">' + v + '</span>';
        h += '</div></div>';
      }
      if (spec.components) {
        h += '<div class="dev-section"><h3>\\ud83e\\udde9 Components</h3><table class="dev-comp-table"><thead><tr><th>Component</th><th>Params</th><th>Trainable</th></tr></thead><tbody>';
        for (const [n,info] of Object.entries(spec.components)) h += '<tr><td>' + n + '</td><td>' + info.params_human + '</td><td>' + info.trainable.toLocaleString() + '</td></tr>';
        h += '</tbody></table></div>';
      }
      if (spec.model_info_json) h += '<div class="dev-section"><h3>\\ud83d\\udcdd model_info.json</h3><div class="dev-json">' + JSON.stringify(spec.model_info_json, null, 2) + '</div></div>';
      if (spec.inputs) h += '<div class="dev-section"><h3>ONNX I/O</h3><div class="dev-json">' + JSON.stringify({inputs:spec.inputs,outputs:spec.outputs}, null, 2) + '</div></div>';
      el.innerHTML = h;
    } catch (e) { $('#model-spec-content').innerHTML = '<div class="dev-placeholder" style="color:var(--red)">Failed: ' + e.message + '</div>'; }
    finally { lsb.disabled = false; lsb.textContent = '\\ud83d\\udccb Load Specification'; }
  });

  // Image metadata analyzer
  const metaDz = $('#meta-dropzone'), metaFi = $('#meta-file-input');
  if (metaDz) {
    metaDz.addEventListener('click', () => metaFi.click());
    metaDz.addEventListener('dragover', e => { e.preventDefault(); metaDz.classList.add('drag-over'); });
    metaDz.addEventListener('dragleave', () => metaDz.classList.remove('drag-over'));
    metaDz.addEventListener('drop', e => { e.preventDefault(); metaDz.classList.remove('drag-over'); if (e.dataTransfer.files.length) analyzeImageMeta(e.dataTransfer.files[0]); });
    metaFi.addEventListener('change', () => { if (metaFi.files.length) analyzeImageMeta(metaFi.files[0]); });
  }
  async function analyzeImageMeta(file) {
    if (!file.type.startsWith('image/')) return;
    const form = new FormData(); form.append('image', file); form.append('filename', file.name);
    try {
      const res = await fetch('/api/image-info', { method: 'POST', body: form });
      const meta = await res.json();
      const el = $('#image-meta-content'); el.classList.remove('hidden');
      let h = '<div class="meta-grid">';
      h += '<div class="meta-card"><h4>\\ud83d\\udcd0 Dimensions</h4><div class="dev-kv"><span class="k">Width</span><span class="v">' + meta.width + 'px</span><span class="k">Height</span><span class="v">' + meta.height + 'px</span><span class="k">Aspect Ratio</span><span class="v">' + (meta.aspect_ratio_simplified||meta.aspect_ratio) + '</span><span class="k">Megapixels</span><span class="v">' + meta.megapixels + ' MP</span><span class="k">Orientation</span><span class="v">' + meta.orientation + '</span></div></div>';
      h += '<div class="meta-card"><h4>\\ud83c\\udfa8 Color Info</h4><div class="dev-kv"><span class="k">Color Space</span><span class="v">' + meta.color_space + '</span><span class="k">Mode</span><span class="v">' + meta.mode + '</span><span class="k">Channels</span><span class="v">' + meta.channels + ' (' + (meta.bands||[]).join(', ') + ')</span><span class="k">Bit Depth</span><span class="v">' + meta.bit_depth + '-bit</span><span class="k">Unique Colors</span><span class="v">' + meta.unique_colors + '</span></div></div>';
      h += '<div class="meta-card"><h4>\\ud83d\\udcbe File Info</h4><div class="dev-kv"><span class="k">Format</span><span class="v">' + meta.format + '</span><span class="k">Size</span><span class="v">' + meta.file_size_kb + ' KB (' + meta.file_size_bytes + ' bytes)</span>';
      if (meta.dpi_x) h += '<span class="k">DPI</span><span class="v">' + meta.dpi_x + ' \\u00d7 ' + meta.dpi_y + '</span>';
      h += '<span class="k">Filename</span><span class="v">' + file.name + '</span></div></div>';
      const ek = Object.keys(meta.exif || {});
      if (ek.length > 0) h += '<div class="meta-card" style="grid-column:1/-1"><h4>\\ud83d\\udcf7 EXIF Data (' + ek.length + ' tags)</h4><div class="dev-json" style="max-height:200px">' + JSON.stringify(meta.exif, null, 2) + '</div></div>';
      h += '</div>';
      el.innerHTML = h;
    } catch (e) { const el = $('#image-meta-content'); el.classList.remove('hidden'); el.innerHTML = '<div class="dev-placeholder" style="color:var(--red)">Failed: ' + e.message + '</div>'; }
  }

  // Presenter config controls
  const mbr = $('#cfg-min-breed'), msr = $('#cfg-min-species');
  if (mbr) mbr.addEventListener('input', () => { $('#cfg-min-breed-val').textContent = mbr.value + '%'; });
  if (msr) msr.addEventListener('input', () => { $('#cfg-min-species-val').textContent = msr.value + '%'; });
  const scb = $('#save-presenter-cfg-btn');
  if (scb) scb.addEventListener('click', async () => {
    const cfg = {
      hide_species_confidence: $('#cfg-hide-species-conf').checked,
      hide_top5_list: $('#cfg-hide-top5').checked,
      hide_info_footer: $('#cfg-hide-footer').checked,
      hide_batch_summary_stats: $('#cfg-hide-batch-stats').checked,
      min_breed_confidence_pct: parseFloat($('#cfg-min-breed').value),
      min_species_confidence_pct: parseFloat($('#cfg-min-species').value),
    };
    try {
      await fetch('/api/presenter-config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cfg) });
      presenterConfig = cfg;
      $('#cfg-save-status').textContent = '\\u2705 Saved! Config persists across restarts.';
      $('#cfg-save-status').style.color = 'var(--green)';
      setTimeout(() => { $('#cfg-save-status').textContent = ''; }, 3000);
    } catch (e) { $('#cfg-save-status').textContent = '\\u274c Save failed: ' + e.message; $('#cfg-save-status').style.color = 'var(--red)'; }
  });

  // Session logs
  const rlb = $('#refresh-logs-btn');
  if (rlb) {
    function loadLogs() {
      fetch('/api/logs').then(r => r.text()).then(txt => {
        const el = $('#session-log-content');
        el.innerHTML = '<pre style="font-size:0.78rem;color:var(--text-muted);white-space:pre-wrap;word-break:break-all;margin:0">' + (txt || '(empty)') + '</pre>';
        el.scrollTop = el.scrollHeight;
      }).catch(() => {});
    }
    rlb.addEventListener('click', loadLogs);
    loadLogs();
  }
}
"""


def run_server(manager, port=8501, mode="dev"):
    """Run a minimal HTTP server with the GUI and prediction API."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    html_content = build_html(mode=mode)

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

            elif self.path == "/api/dev-info":
                info = {"cattle_classes": manager.class_maps.get("cattle"),
                        "buffalo_classes": manager.class_maps.get("buffalo")}
                self._respond(200, "application/json", json.dumps(info).encode())

            elif self.path.startswith("/api/model-spec"):
                from urllib.parse import urlparse, parse_qs
                qs = parse_qs(urlparse(self.path).query)
                model_name = qs.get("model", [""])[0]
                if model_name not in manager.models:
                    self._respond(404, "application/json",
                                  json.dumps({"error": "Model not found"}).encode())
                    return
                try:
                    model = manager._load_model(model_name)
                    spec = get_model_spec(model, model_name, manager.models[model_name])
                    self._respond(200, "application/json", json.dumps(spec).encode())
                except Exception as e:
                    self._respond(500, "application/json",
                                  json.dumps({"error": str(e)}).encode())

            elif self.path == "/api/presenter-config":
                cfg = load_presenter_config()
                self._respond(200, "application/json", json.dumps(cfg).encode())

            elif self.path == "/api/logs":
                if logger:
                    content = logger.get_log_contents()
                else:
                    content = "(logger not initialized)"
                self._respond(200, "text/plain", content.encode())

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

                if logger:
                    logger.info(f"Model selected: {model_name}")

                result = manager.predict(image_bytes, model_name)
                result["filename"] = filename

                # In present mode, strip dev-only fields
                if mode == "present":
                    result.pop("_raw_binary_probs", None)
                    result.pop("_raw_breed_probs_top10", None)
                    result.pop("inference_time_ms", None)
                    result.pop("total_time_ms", None)

                self._respond(200, "application/json", json.dumps(result).encode())

            elif self.path == "/api/export-odt":
                if mode == "present":
                    self._respond(403, "application/json",
                                  json.dumps({"error": "Export is disabled in presenter mode"}).encode())
                    return

                content_length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    results = payload.get("results", {})
                    odt_mode = payload.get("mode", "single")

                    if logger:
                        logger.info(f"ODT export requested ({odt_mode} mode)")

                    odt_bytes = generate_odt_report(results, mode=odt_mode)
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

            elif self.path == "/api/image-info":
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self._respond(400, "application/json",
                                  json.dumps({"error": "Expected multipart/form-data"}).encode())
                    return
                parts_data, parts_text = _parse_multipart(self)
                image_bytes = parts_data.get("image")
                if not image_bytes:
                    self._respond(400, "application/json",
                                  json.dumps({"error": "No image provided"}).encode())
                    return
                meta = extract_image_metadata(image_bytes)
                if logger:
                    logger.info(f"Image metadata extracted: {meta.get('width', '?')}Ã—{meta.get('height', '?')} {meta.get('format', '?')}")
                self._respond(200, "application/json", json.dumps(meta).encode())

            elif self.path == "/api/presenter-config":
                content_length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(content_length)
                try:
                    cfg = json.loads(raw.decode("utf-8"))
                    save_presenter_config(cfg)
                    if logger:
                        logger.info(f"Presenter config saved: {cfg}")
                    self._respond(200, "application/json",
                                  json.dumps({"status": "saved"}).encode())
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
    print(f"\n  ðŸŒ Model Tester GUI running at http://localhost:{port}")
    print(f"  Mode: {'ðŸ› ï¸  Developer' if mode == 'dev' else 'ðŸŽ¤ Presenter'}")
    print(f"  Press Ctrl+C to stop\n")
    server.serve_forever()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    global logger

    parser = argparse.ArgumentParser(
        description="ðŸ„ Breed Classifier â€” Model Tester GUI")
    parser.add_argument("--port", type=int, default=8501,
                        help="port to serve on (default: 8501)")
    parser.add_argument("--no-browser", action="store_true",
                        help="don't auto-open browser")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dev", action="store_true", default=True,
                            help="developer mode with extra tools (default)")
    mode_group.add_argument("--present", action="store_true",
                            help="presenter mode â€” clean, polished view")
    args = parser.parse_args()

    mode = "present" if args.present else "dev"

    # Start session logger
    logger = SessionLogger()

    print("\n" + "=" * 60)
    print(f"  ðŸ„ Breed Classifier â€” Model Tester GUI")
    print(f"  Mode: {'ðŸ› ï¸  Developer' if mode == 'dev' else 'ðŸŽ¤ Presenter'}")
    print("=" * 60)

    logger.info(f"Application started (mode={mode}, port={args.port})")

    manager = ModelManager()
    models = manager.list_models()

    if not models:
        print("\n  âŒ No models found!")
        print("  Train a model first:")
        print("    python local_train.py --smoke-test")
        print("    python local_train.py --quarter-data")
        logger.error("No models found â€” exiting")
        logger.stop()
        return 1

    print(f"\n  Device: {manager.device}")
    print(f"  Models found: {len(models)}")
    for m in models:
        print(f"    â€¢ {m['name']} ({m['backbone']}, {m['size_mb']} MB)")

    if not manager.class_maps:
        print("\n  âš ï¸  Warning: No class maps found in data/splits/")
        print("  Breed names will show as class indices.")
        logger.warning("No class maps found in data/splits/")

    logger.info(f"Session log: {logger.log_path}")

    # Auto-open browser
    if not args.no_browser:
        url = f"http://localhost:{args.port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        run_server(manager, port=args.port, mode=mode)
    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        logger.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
