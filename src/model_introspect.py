"""Self-describing model loading and metadata embedding for ``.pt`` / ``.onnx``.

Why this module exists
----------------------
Historically the GUI (``test_model.py``) and the Flask API (``server/``) needed
external JSON files — ``cattle_classes.json``, ``buffalo_classes.json``,
``model_info.json`` and ``outputs/ood/*.json`` — beside every model.  A model
copied on its own therefore either failed to load or silently mislabelled its
classes, and a hard-coded ``IMAGE_SIZE`` / caller-side ImageNet
``Normalize`` meant a mobile ONNX graph (normalization baked in) was
normalized twice.

This module makes a model file the single source of truth:

* ``inspect_and_load_pt`` / ``inspect_and_load_onnx`` read architecture,
  resolution, class maps, preprocessing mode and OOD thresholds **from the
  file itself**.  Embedded metadata wins; tensor / graph introspection is the
  fallback, so pre-existing artifacts work with zero external JSON.
* ``embed_pt_metadata`` / ``embed_onnx_metadata`` permanently write that
  metadata into new or existing files.
* ``LoadedModelBundle`` hands the callers everything they need (model/session,
  transform, inverse class maps, OOD config, a spec summary for the Dev
  Console) and can be used for PyTorch and ONNX interchangeably.

Metadata schema (version 1.0)
-----------------------------
``model_meta`` on a ``.pt`` checkpoint / ``custom_metadata_map`` on an
``.onnx`` graph::

    model_meta_version : "1.0"
    backbone           : "lite2" | "lite4"
    attention          : "cbam" | "se"
    feature_dim        : 1280
    binary_dim         : 512
    breed_dim          : 384
    cosine_head        : false
    num_cattle         : 50
    num_buffalo        : 18
    image_size         : 260
    eval_resize        : 260
    pad_to_square      : true
    imagenet_mean      : [0.485, 0.456, 0.406]
    imagenet_std       : [0.229, 0.224, 0.225]
    normalization_baked: false      # true -> feed [0, 1] RGB, no Normalize()
    normalization_mode : "caller" | "baked"
    input_dtype        : "tensor(float)" | "tensor(float16)" | "float32" | ...
    cattle_classes     : {"gir": 0, ...}
    buffalo_classes    : {"murrah": 0, ...}
    ood                : {"enabled": true, "energy_threshold": ...,
                          "msp_threshold": ..., "temperature": ...,
                          "binary_confidence_floor": ...}
    trait_vocab        : {...}      # optional
"""

from __future__ import annotations

import copy
import datetime
import glob
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .config import (
    BREED_DIM,
    BINARY_DIM,
    EXPECTED_BUFFALO_BREEDS,
    EXPECTED_CATTLE_BREEDS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_BUFFALO_BREEDS,
    NUM_CATTLE_BREEDS,
    OOD_BINARY_CONFIDENCE_FLOOR,
    OOD_DIR,
    OOD_ENABLED,
    OOD_ENERGY_TEMPERATURE,
    OOD_ENERGY_THRESHOLD,
    OOD_MSP_THRESHOLD,
    PORTABLE_EXPORT_DIR,
    SPLIT_DIR,
    TRAIT_FILE,
)

MODEL_META_VERSION = "1.0"
PT_META_KEY = "model_meta"
TORCHSCRIPT_META_FILE = "model_meta.json"

_NUMERIC_SUFFIXES = (
    ".weight_fake_quant", ".activation_post_process",
    ".fake_quant_enabled", ".observer_enabled", ".scale", ".zero_point",
    ".min_val", ".max_val", ".eps",
)


# ---------------------------------------------------------------------------
# State-dict hygiene and tensor introspection
# ---------------------------------------------------------------------------

def sanitize_state_dict(state: Any) -> Dict[str, torch.Tensor]:
    """Strip compile/DataParallel prefixes and QAT/fused-BN observer keys.

    Mirrors the cleanup used by ``src/export.py`` so exported and inspected
    checkpoints agree on the same key set.
    """
    if not isinstance(state, dict):
        return {}
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        k = str(key)
        for prefix in ("module.", "_orig_mod."):
            while k.startswith(prefix):
                k = k[len(prefix):]
        if ".weight_fake_quant" in k or ".activation_post_process" in k:
            continue
        if k.endswith(_NUMERIC_SUFFIXES):
            continue
        cleaned[k] = value
    return cleaned


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Return the raw ``state_dict`` from any supported checkpoint shape."""
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict"):
            if isinstance(ckpt.get(key), dict):
                return ckpt[key]
        # A plain ``{tensor_name: tensor}`` checkpoint.
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    return {}


def _last_head_index(state: Dict[str, Any], prefix: str) -> Optional[int]:
    indexes: List[int] = []
    for key in state:
        if not key.startswith(prefix + "."):
            continue
        parts = key.split(".")
        if len(parts) > 2 and parts[1].isdigit():
            indexes.append(int(parts[1]))
    return max(indexes) if indexes else None


def inspect_state_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the architecture purely from tensor shapes (never the filename)."""
    state = sanitize_state_dict(state)
    if not state:
        raise ValueError("checkpoint contains no tensors")

    spec: Dict[str, Any] = {
        "backbone": None,
        "attention": None,
        "feature_dim": None,
        "binary_dim": None,
        "breed_dim": None,
        "num_cattle": None,
        "num_buffalo": None,
        "cosine_head": False,
        "has_trait_heads": False,
        "final_head_index": None,
    }

    # Backbone: first MBConv projection width is 16 (lite2) or 24 (lite4).
    probe = "backbone.blocks.0.0._project_conv.weight"
    if probe in state:
        width = int(state[probe].shape[0])
        spec["backbone"] = "lite2" if width == 16 else ("lite4" if width == 24 else None)
    if spec["backbone"] is None:
        # Fallback: stage-6 width 352 (lite2) / 448 (lite4).
        for key in state:
            if key.startswith("backbone.blocks.6.") and key.endswith("_project_conv.weight"):
                width = int(state[key].shape[0])
                spec["backbone"] = "lite4" if width >= 400 else "lite2"
                break

    # Attention: CBAM adds a spatial conv; SE only has the channel MLP.
    if "attention.spatial_attention.conv.weight" in state:
        spec["attention"] = "cbam"
    elif any(k.startswith("attention.") for k in state):
        spec["attention"] = "se"

    if "binary_head.0.weight" in state:
        spec["binary_dim"] = int(state["binary_head.0.weight"].shape[0])
        spec["feature_dim"] = int(state["binary_head.0.weight"].shape[1])
    elif "cattle_head.0.weight" in state:
        spec["feature_dim"] = int(state["cattle_head.0.weight"].shape[1])

    if "cattle_head.0.weight" in state:
        spec["breed_dim"] = int(state["cattle_head.0.weight"].shape[0])

    fi = _last_head_index(state, "cattle_head")
    spec["final_head_index"] = fi
    if fi is not None:
        final_key = f"cattle_head.{fi}.weight"
        if final_key in state:
            spec["num_cattle"] = int(state[final_key].shape[0])
            spec["cosine_head"] = f"cattle_head.{fi}.bias" not in state
    bi = _last_head_index(state, "buffalo_head")
    if bi is not None:
        final_key = f"buffalo_head.{bi}.weight"
        if final_key in state:
            spec["num_buffalo"] = int(state[final_key].shape[0])
            if f"buffalo_head.{bi}.bias" not in state:
                spec["cosine_head"] = True

    spec["has_trait_heads"] = any(k.startswith("trait_heads.heads.") for k in state)
    return spec


# ---------------------------------------------------------------------------
# Class maps
# ---------------------------------------------------------------------------

def _normalise_class_map(value: Any) -> Dict[str, int]:
    """Accept ``{name: idx}``, ``{idx: name}``, a list/tuple of names, or JSON."""
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {}
    result: Dict[str, int] = {}
    if isinstance(value, dict):
        numeric_keys = all(str(k).isdigit() for k in value)
        if numeric_keys:
            for idx, name in value.items():
                result[str(name)] = int(idx)
        else:
            for name, idx in value.items():
                try:
                    result[str(name)] = int(idx)
                except (TypeError, ValueError):
                    continue
        return result
    if isinstance(value, (list, tuple)):
        return {str(name): i for i, name in enumerate(value)}
    return {}


def _expected_maps(num_cattle: Optional[int],
                   num_buffalo: Optional[int]) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
    if num_cattle != len(EXPECTED_CATTLE_BREEDS):
        return None
    if num_buffalo != len(EXPECTED_BUFFALO_BREEDS):
        return None
    return (
        {name: i for i, name in enumerate(EXPECTED_CATTLE_BREEDS)},
        {name: i for i, name in enumerate(EXPECTED_BUFFALO_BREEDS)},
    )


def _synthetic_map(prefix: str, count: Optional[int]) -> Dict[str, int]:
    return {f"{prefix}_{i}": i for i in range(int(count or 0))}


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _read_label_file(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    except Exception:
        return []


def _load_maps_from_dir(directory: str,
                        num_cattle: Optional[int],
                        num_buffalo: Optional[int]) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
    if not directory or not os.path.isdir(directory):
        return None
    cattle = _normalise_class_map(_read_json(os.path.join(directory, "cattle_classes.json")))
    buffalo = _normalise_class_map(_read_json(os.path.join(directory, "buffalo_classes.json")))
    if not cattle:
        cattle = _normalise_class_map(_read_label_file(os.path.join(directory, "labels_cattle.txt")))
    if not buffalo:
        buffalo = _normalise_class_map(_read_label_file(os.path.join(directory, "labels_buffalo.txt")))
    if not cattle or not buffalo:
        return None
    if num_cattle is not None and len(cattle) != num_cattle:
        return None
    if num_buffalo is not None and len(buffalo) != num_buffalo:
        return None
    return cattle, buffalo


def resolve_class_maps(
    num_cattle: Optional[int],
    num_buffalo: Optional[int],
    embedded: Optional[Dict[str, Any]] = None,
    search_dirs: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, int], Dict[str, int], str]:
    """Resolve cattle/buffalo maps without requiring external JSON.

    Priority: embedded metadata -> canonical config lists (count match) ->
    a count-matching JSON/label file in ``search_dirs`` -> synthetic names.
    """
    embedded = embedded or {}
    cattle = _normalise_class_map(embedded.get("cattle_classes"))
    buffalo = _normalise_class_map(embedded.get("buffalo_classes"))
    if cattle and buffalo:
        if ((num_cattle is None or len(cattle) == num_cattle)
                and (num_buffalo is None or len(buffalo) == num_buffalo)):
            return cattle, buffalo, "embedded"

    expected = _expected_maps(num_cattle, num_buffalo)
    if expected is not None:
        return expected[0], expected[1], "config:EXPECTED_*_BREEDS"

    for directory in search_dirs or ():
        found = _load_maps_from_dir(directory, num_cattle, num_buffalo)
        if found is not None:
            return found[0], found[1], directory

    # Last structured fallback: a portable bundle with matching head widths.
    if num_cattle is not None and num_buffalo is not None:
        for bundle_dir in sorted(glob.glob(os.path.join(PORTABLE_EXPORT_DIR, "*"))):
            if not os.path.isdir(bundle_dir):
                continue
            found = _load_maps_from_dir(bundle_dir, num_cattle, num_buffalo)
            if found is not None:
                return found[0], found[1], bundle_dir

    return (_synthetic_map("cattle", num_cattle or NUM_CATTLE_BREEDS),
            _synthetic_map("buffalo", num_buffalo or NUM_BUFFALO_BREEDS),
            "synthetic")


# ---------------------------------------------------------------------------
# Preprocessing + OOD defaults
# ---------------------------------------------------------------------------

def default_ood_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return config defaults merged with embedded/CLI overrides."""
    config = {
        "enabled": bool(OOD_ENABLED),
        "energy_temperature": float(OOD_ENERGY_TEMPERATURE),
        "temperature": float(OOD_ENERGY_TEMPERATURE),
        "energy_threshold": float(OOD_ENERGY_THRESHOLD),
        "msp_threshold": float(OOD_MSP_THRESHOLD),
        "binary_confidence_floor": float(OOD_BINARY_CONFIDENCE_FLOOR),
    }
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    if "temperature" not in (overrides or {}) and "energy_temperature" in (overrides or {}):
        config["temperature"] = config["energy_temperature"]
    return config


def find_ood_json(model_path: str) -> Optional[str]:
    """Find a per-model OOD JSON next to the model or under ``outputs/ood``."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    clean = re.sub(r"^\[.*?\]\s*", "", stem)
    clean = re.sub(r"\s*\(ONNX\)$", "", clean, flags=re.IGNORECASE)
    candidates = [
        os.path.join(os.path.dirname(model_path), f"{clean}.json"),
        os.path.join(os.path.dirname(model_path), "ood_thresholds.json"),
        os.path.join(OOD_DIR, f"{clean}.json"),
        os.path.join(OOD_DIR, f"{clean}_ood.json"),
        os.path.join(OOD_DIR, "ood_thresholds.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _ood_from_file(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    return {
        "enabled": True,
        "energy_temperature": data.get("energy_temperature", data.get("temperature")),
        "temperature": data.get("energy_temperature", data.get("temperature")),
        "energy_threshold": data.get("energy_threshold"),
        "msp_threshold": data.get("msp_threshold"),
        "binary_confidence_floor": data.get("binary_confidence_floor"),
        "calibrated_at": data.get("calibrated_at"),
        "percentile": data.get("percentile"),
        "calibrated_on_samples": data.get("calibrated_on_samples"),
        "source_file": path,
    }


def default_preprocess(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from .config import INFERENCE_PAD_TO_SQUARE

    config = {
        "image_size": int(IMAGE_SIZE),
        "eval_resize": int(IMAGE_SIZE),
        "pad_to_square": bool(INFERENCE_PAD_TO_SQUARE),
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "normalization_baked": False,
        "normalization_mode": "caller",
    }
    for key, value in (overrides or {}).items():
        if value is not None:
            config[key] = value
    config["normalization_mode"] = "baked" if config.get("normalization_baked") else "caller"
    return config


def build_transform(image_size: int = IMAGE_SIZE,
                    normalization_baked: bool = False,
                    pad_to_square: Optional[bool] = None,
                    mean: Optional[Iterable[float]] = None,
                    std: Optional[Iterable[float]] = None,
                    eval_resize: Optional[int] = None):
    """Build the exact preprocessing chain the model expects.

    ``normalization_baked=True`` (mobile ONNX / TFLite) stops at ``ToTensor``
    so the graph receives ``[0, 1]`` and applies mean/std internally.  This is
    the fix for the historical double-normalization bug.
    """
    from torchvision import transforms

    from .preprocess import PadToSquare

    if pad_to_square is None:
        from .config import INFERENCE_PAD_TO_SQUARE
        pad_to_square = INFERENCE_PAD_TO_SQUARE

    size = int(image_size or IMAGE_SIZE)
    blocks: List[Any] = []
    if pad_to_square:
        blocks.append(PadToSquare(size))
    else:
        resize = int(eval_resize or size)
        if resize != size:
            blocks.extend([transforms.Resize(resize), transforms.CenterCrop(size)])
        else:
            blocks.extend([transforms.Resize(size), transforms.CenterCrop(size)])
    blocks.append(transforms.ToTensor())
    if not normalization_baked:
        blocks.append(transforms.Normalize(
            mean=list(mean or IMAGENET_MEAN), std=list(std or IMAGENET_STD)))
    return transforms.Compose(blocks)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class LoadedModelBundle:
    """Everything a caller needs to run one model file, with no JSON sidecars."""

    engine: str                       # "pt" | "onnx" | "torchscript"
    model_or_session: Any
    path: str
    transform: Any
    cattle_map: Dict[str, int]
    buffalo_map: Dict[str, int]
    ood_config: Dict[str, Any]
    spec: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    class_map_source: str = "synthetic"
    normalization_baked: bool = False
    image_size: int = IMAGE_SIZE
    input_dtype: str = "float32"
    device: str = "cpu"
    onnx_input_name: Optional[str] = None
    onnx_output_names: Optional[List[str]] = None

    @property
    def cattle_inv_map(self) -> Dict[int, str]:
        return {idx: name for name, idx in self.cattle_map.items()}

    @property
    def buffalo_inv_map(self) -> Dict[int, str]:
        return {idx: name for name, idx in self.buffalo_map.items()}

    @property
    def num_cattle(self) -> int:
        return len(self.cattle_map)

    @property
    def num_buffalo(self) -> int:
        return len(self.buffalo_map)

    @property
    def is_onnx(self) -> bool:
        return self.engine == "onnx"

    @property
    def ood_detector(self):
        """OODDetector built from the embedded thresholds (no JSON lookup)."""
        from .model import OODDetector
        return OODDetector.from_embedded(self.ood_config)

    def spec_summary(self) -> Dict[str, Any]:
        """Compact, JSON-safe description for the Dev Console / APIs."""
        summary = {
            "engine": self.engine,
            "path": self.path,
            "class_map_source": self.class_map_source,
            "num_cattle": self.num_cattle,
            "num_buffalo": self.num_buffalo,
            "preprocessing": {
                "image_size": self.image_size,
                "normalization_baked": self.normalization_baked,
                "normalization_mode": "baked" if self.normalization_baked else "caller",
                "input_dtype": self.input_dtype,
                "transform": repr(self.transform),
            },
            "ood": copy.deepcopy(self.ood_config),
            "class_maps": {
                "cattle": dict(self.cattle_map),
                "buffalo": dict(self.buffalo_map),
            },
        }
        if self.onnx_input_name:
            summary["onnx_input_name"] = self.onnx_input_name
        if self.onnx_output_names:
            summary["onnx_output_names"] = list(self.onnx_output_names)
        summary.update(copy.deepcopy(self.spec or {}))
        return summary


# ---------------------------------------------------------------------------
# PyTorch loading
# ---------------------------------------------------------------------------

def read_pt_metadata(path: str) -> Dict[str, Any]:
    """Read embedded metadata without constructing the network."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return _read_torchscript_metadata(path)
    if isinstance(ckpt, dict):
        meta = ckpt.get(PT_META_KEY) or ckpt.get("metadata") or {}
        return copy.deepcopy(meta) if isinstance(meta, dict) else {}
    return _read_torchscript_metadata(path)


def _read_torchscript_metadata(path: str) -> Dict[str, Any]:
    try:
        extra = {TORCHSCRIPT_META_FILE: ""}
        torch.jit.load(path, map_location="cpu", _extra_files=extra)
        raw = extra.get(TORCHSCRIPT_META_FILE) or ""
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def _infer_torchscript_spec(module: Any) -> Dict[str, Any]:
    """Probe a traced module so class widths are still discovered."""
    spec: Dict[str, Any] = {"cosine_head": False, "has_trait_heads": False}
    try:
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = module(dummy)
        if isinstance(out, dict):
            tensors = [out.get("binary"), out.get("cattle"), out.get("buffalo")]
        elif isinstance(out, (tuple, list)):
            tensors = list(out)[:3]
        else:
            tensors = [out]
        if len(tensors) >= 3 and tensors[0] is not None:
            spec["binary_dim"] = int(tensors[0].shape[-1])
            spec["num_cattle"] = int(tensors[1].shape[-1])
            spec["num_buffalo"] = int(tensors[2].shape[-1])
        # Walk the module for the backbone / attention footprint.
        for name, _ in module.named_modules():
            if "spatial_attention" in name:
                spec["attention"] = "cbam"
            elif name.endswith("attention") and "attention" not in spec:
                spec["attention"] = "se"
        for name, param in module.named_parameters():
            if "backbone.blocks.0.0._project_conv.weight" in name:
                spec["backbone"] = "lite2" if int(param.shape[0]) == 16 else "lite4"
            if "binary_head.0.weight" in name:
                spec["feature_dim"] = int(param.shape[1])
                spec["binary_dim"] = int(param.shape[0])
    except Exception:
        pass
    return spec


def _adapt_legacy_heads(model: Any, state: Dict[str, Any]) -> List[str]:
    """Rebuild head modules whose layer layout predates the current model.

    Early exports (e.g. the v17 portable bundle) used a 7-layer binary head
    ``Linear-BN-ReLU-Drop-Linear-ReLU-Linear`` instead of today's 9-layer
    version.  The weights are valid, only the module indices differ, so we
    recreate the matching ``Sequential`` before loading.
    """
    import torch.nn as nn

    from .config import DROPOUT

    linear = []
    for key, value in state.items():
        if (key.startswith("binary_head.") and key.endswith(".weight")
                and torch.is_tensor(value) and value.dim() == 2):
            try:
                linear.append(int(key.split(".")[1]))
            except (IndexError, ValueError):
                continue
    indices = sorted(linear)
    if not indices:
        return []
    final = indices[-1]
    if final == 8:
        return []  # current layout
    if final == 6 and 4 in indices and "binary_head.0.weight" in state:
        feat = int(state["binary_head.0.weight"].shape[1])
        h1 = int(state["binary_head.0.weight"].shape[0])
        h2 = int(state["binary_head.4.weight"].shape[0])
        has_bias = "binary_head.6.bias" in state
        model.binary_head = nn.Sequential(
            nn.Linear(feat, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 2, bias=has_bias),
        )
        return [f"binary_head legacy layout ({final + 1} layers) rebuilt"]
    return []


def inspect_and_load_pt(path: str, device: Any = "cpu") -> LoadedModelBundle:
    """Load a ``.pt`` checkpoint and derive every runtime parameter from it."""
    from .model import BreedClassifier, attach_trait_heads

    meta = read_pt_metadata(path)
    raw = None
    scripted = False
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        try:
            raw = torch.jit.load(path, map_location="cpu")
            scripted = True
        except Exception as exc:
            raise RuntimeError(f"cannot load '{path}' as a PyTorch model: {exc}") from exc

    # A full nn.Module (TorchScript or an exported pickle) is runnable as-is.
    if isinstance(raw, torch.nn.Module) or scripted:
        module = raw
        module.eval()
        spec = _infer_torchscript_spec(module)
        engine = "torchscript"
        num_cattle = spec.get("num_cattle") or meta.get("num_cattle")
        num_buffalo = spec.get("num_buffalo") or meta.get("num_buffalo")
        pre = default_preprocess(meta)
        maps = resolve_class_maps(num_cattle, num_buffalo, embedded=meta,
                                  search_dirs=_search_dirs(path))
        bundle = _make_bundle(
            engine=engine, obj=module, path=path, spec=spec, meta=meta,
            cattle_map=maps[0], buffalo_map=maps[1], source=maps[2],
            num_cattle=num_cattle, num_buffalo=num_buffalo, pre=pre,
            device=str(device),
        )
        _move_to_device(module, device)
        return bundle

    state = sanitize_state_dict(extract_state_dict(raw))
    if not state:
        raise ValueError(f"'{path}' is not a state_dict, checkpoint dict or TorchScript module")

    spec = inspect_state_dict(state)
    for key in ("backbone", "attention", "feature_dim", "binary_dim", "breed_dim",
                "cosine_head", "num_cattle", "num_buffalo", "has_trait_heads"):
        if meta.get(key) is not None:
            spec[key] = meta[key]
    backbone = spec.get("backbone") or "lite2"
    attention = spec.get("attention") or "cbam"

    model = BreedClassifier(
        backbone=backbone,
        attention=attention,
        cosine_head=bool(spec.get("cosine_head", False)),
        binary_dim=int(spec.get("binary_dim") or BINARY_DIM),
        breed_dim=int(spec.get("breed_dim") or BREED_DIM),
        num_cattle=int(spec.get("num_cattle") or NUM_CATTLE_BREEDS),
        num_buffalo=int(spec.get("num_buffalo") or NUM_BUFFALO_BREEDS),
    )

    model_state = model.state_dict()
    layout_notes = _adapt_legacy_heads(model, state)
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state.items():
        if key not in model_state:
            continue
        if not torch.is_tensor(value):
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped.append(key)
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing_real = [k for k in missing
                    if not k.startswith(("projection_head.", "trait_heads."))]
    if missing_real:
        raise RuntimeError(
            f"checkpoint {os.path.basename(path)} does not match "
            f"{backbone}+{attention}: missing {len(missing_real)} keys "
            f"(e.g. {missing_real[:5]})")

    if spec.get("has_trait_heads"):
        try:
            attach_trait_heads(model, state)
        except Exception:
            pass

    model.eval()
    _move_to_device(model, device)
    spec["skipped_tensors"] = skipped
    if layout_notes:
        spec["layout_notes"] = layout_notes

    pre = default_preprocess(meta)
    maps = resolve_class_maps(spec.get("num_cattle"), spec.get("num_buffalo"),
                              embedded=meta, search_dirs=_search_dirs(path))
    return _make_bundle(
        engine="pt", obj=model, path=path, spec=spec, meta=meta,
        cattle_map=maps[0], buffalo_map=maps[1], source=maps[2],
        num_cattle=spec.get("num_cattle"), num_buffalo=spec.get("num_buffalo"),
        pre=pre, device=str(device),
    )


def _search_dirs(path: str) -> List[str]:
    """Directories that may legally contain maps for this specific model."""
    directory = os.path.dirname(os.path.abspath(path))
    dirs = [directory]
    # Portable bundles keep their maps beside model.pt; checkpoints use splits.
    if os.path.basename(directory).startswith(("lite2", "lite4")):
        dirs.append(directory)
    dirs.append(SPLIT_DIR)
    dirs.append(os.path.join(PORTABLE_EXPORT_DIR))
    # Deduplicate while preserving order.
    seen, ordered = set(), []
    for item in dirs:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _move_to_device(model: Any, device: Any) -> None:
    try:
        model.to(device)
    except Exception:
        pass


def _make_bundle(*, engine: str, obj: Any, path: str, spec: Dict[str, Any],
                 meta: Dict[str, Any], cattle_map: Dict[str, int],
                 buffalo_map: Dict[str, int], source: str,
                 num_cattle: Optional[int], num_buffalo: Optional[int],
                 pre: Dict[str, Any], device: str) -> LoadedModelBundle:
    # Class maps must agree with the tensor widths; never silently pair a
    # 50-class map with a 42-class head.
    if num_cattle is not None and len(cattle_map) != num_cattle:
        cattle_map = _synthetic_map("cattle", num_cattle)
    if num_buffalo is not None and len(buffalo_map) != num_buffalo:
        buffalo_map = _synthetic_map("buffalo", num_buffalo)

    clean_spec = {k: v for k, v in (spec or {}).items() if v is not None}
    clean_spec.update({
        "feature_dim": int(clean_spec.get("feature_dim") or meta.get("feature_dim") or 1280),
        "binary_dim": int(clean_spec.get("binary_dim") or meta.get("binary_dim") or BINARY_DIM),
        "breed_dim": int(clean_spec.get("breed_dim") or meta.get("breed_dim") or BREED_DIM),
        "num_cattle": int(num_cattle or len(cattle_map)),
        "num_buffalo": int(num_buffalo or len(buffalo_map)),
        "backbone": clean_spec.get("backbone") or meta.get("backbone") or "lite2",
        "attention": clean_spec.get("attention") or meta.get("attention") or "cbam",
        "cosine_head": bool(clean_spec.get("cosine_head", False)),
        "has_trait_heads": bool(clean_spec.get("has_trait_heads", False)),
    })

    normal_baked = bool(pre.get("normalization_baked"))
    transform = build_transform(
        image_size=pre.get("image_size", IMAGE_SIZE),
        normalization_baked=normal_baked,
        pad_to_square=pre.get("pad_to_square"),
        mean=pre.get("imagenet_mean"),
        std=pre.get("imagenet_std"),
        eval_resize=pre.get("eval_resize"),
    )
    ood = default_ood_config(meta.get("ood"))
    if meta.get("ood"):
        ood["source_json"] = meta["ood"].get("source_file")
        ood["source_file"] = f"embedded:{os.path.basename(path)}"
    else:
        ood["source_file"] = "config:defaults"

    return LoadedModelBundle(
        engine=engine,
        model_or_session=obj,
        path=path,
        transform=transform,
        cattle_map=cattle_map,
        buffalo_map=buffalo_map,
        ood_config=ood,
        spec=clean_spec,
        metadata=copy.deepcopy(meta),
        class_map_source=source,
        normalization_baked=normal_baked,
        image_size=int(pre.get("image_size", IMAGE_SIZE)),
        input_dtype=str(pre.get("input_dtype") or "float32"),
        device=str(device),
    )


# ---------------------------------------------------------------------------
# ONNX loading
# ---------------------------------------------------------------------------

def _parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().strip('"').lower()
    if text in ("true", "1", "yes", "on", "baked"):
        return True
    if text in ("false", "0", "no", "off", "caller"):
        return False
    return None


def _jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def read_onnx_metadata(path: str) -> Dict[str, Any]:
    """Read ``custom_metadata_map`` (ORT) with an ``onnx`` fallback."""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        raw = dict(session.get_modelmeta().custom_metadata_map or {})
    except Exception:
        raw = {}
    if not raw:
        try:
            import onnx
            model = onnx.load(path, load_external_data=False)
            raw = {prop.key: prop.value for prop in model.metadata_props}
        except Exception:
            raw = {}
    return {
        key: (value if key == "model_meta_version" else _jsonish(value))
        for key, value in raw.items()
    }


def _detect_onnx_normalization(path: str, meta: Dict[str, Any]) -> bool:
    """Decide whether mean/std are baked into the graph.

    Explicit metadata wins.  Otherwise inspect initializers/constants for the
    ImageNet mean and std (the mobile wrapper registers exactly those), check
    for a leading ``Sub -> Div`` on the input, and finally fall back to the
    legacy ``*_mobile*`` / ``*_int8*`` filename convention.
    """
    explicit = _parse_bool(meta.get("normalization_baked"))
    if explicit is not None:
        return explicit
    mode = str(meta.get("normalization_mode") or "").lower()
    if mode in ("baked", "caller"):
        return mode == "baked"

    try:
        import onnx
        model = onnx.load(path, load_external_data=False)
        initializers = list(model.graph.initializer)
        for init in initializers:
            try:
                arr = onnx.numpy_helper.to_array(init).ravel().astype(float)
            except Exception:
                continue
            if arr.size != 3:
                continue
            if (abs(arr - 0.485) < 1e-3).all() or (abs(arr - 0.229) < 1e-3).all():
                return True
        first_ops = [node.op_type for node in model.graph.node[:4]]
        if "Sub" in first_ops and "Div" in first_ops:
            return True
    except Exception:
        pass

    name = os.path.basename(path).lower()
    return "mobile" in name or "int8" in name


def _last_int_dim(shape: Any) -> Optional[int]:
    for value in reversed(list(shape or [])):
        if isinstance(value, int):
            return int(value)
    return None


def _match_onnx_outputs(session: Any) -> Tuple[Any, Any, Any]:
    outputs = list(session.get_outputs())
    by_name = {str(o.name).lower(): o for o in outputs}
    binary = by_name.get("binary")
    cattle = by_name.get("cattle")
    buffalo = by_name.get("buffalo")
    if binary is not None and cattle is not None and buffalo is not None:
        return binary, cattle, buffalo

    # Unnamed outputs: width 2 is binary, the largest remaining is cattle.
    remaining = [o for o in outputs if o is not binary and o is not cattle and o is not buffalo]
    if binary is None:
        binary = next((o for o in remaining if _last_int_dim(o.shape) == 2), None)
        remaining = [o for o in remaining if o is not binary]
    remaining.sort(key=lambda o: _last_int_dim(o.shape) or -1, reverse=True)
    if cattle is None and remaining:
        cattle = remaining.pop(0)
    if buffalo is None and remaining:
        buffalo = remaining.pop(0)
    return binary, cattle, buffalo


def inspect_and_load_onnx(path: str, use_cuda: bool = True) -> LoadedModelBundle:
    """Load an ONNX graph and introspect its input/output contract."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. Install it with "
            "`pip install onnxruntime` (or onnxruntime-gpu).") from exc

    providers = ["CPUExecutionProvider"]
    if use_cuda:
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(path, providers=providers)

    meta = read_onnx_metadata(path)
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError(f"ONNX model '{path}' has no inputs")
    input_meta = inputs[0]
    input_shape = list(input_meta.shape or [])
    image_size = _last_int_dim(input_shape[2:4]) or int(meta.get("image_size") or IMAGE_SIZE)
    if len(input_shape) >= 4 and isinstance(input_shape[2], int):
        image_size = int(input_shape[2])
    input_dtype = str(meta.get("input_dtype") or input_meta.type or "tensor(float)")

    binary_out, cattle_out, buffalo_out = _match_onnx_outputs(session)
    num_cattle = _last_int_dim(getattr(cattle_out, "shape", None)) or meta.get("num_cattle")
    num_buffalo = _last_int_dim(getattr(buffalo_out, "shape", None)) or meta.get("num_buffalo")

    baked = _detect_onnx_normalization(path, meta)
    pre = default_preprocess(meta)
    pre["image_size"] = int(image_size)
    pre["normalization_baked"] = baked
    pre["input_dtype"] = input_dtype
    pre["eval_resize"] = int(meta.get("eval_resize") or image_size)

    maps = resolve_class_maps(num_cattle, num_buffalo, embedded=meta,
                              search_dirs=_search_dirs(path))
    spec = {
        "backbone": meta.get("backbone"),
        "attention": meta.get("attention"),
        "num_cattle": num_cattle,
        "num_buffalo": num_buffalo,
        "input_shape": input_shape,
        "normalization_baked": baked,
        "onnx_outputs": {
            "binary": getattr(binary_out, "name", None),
            "cattle": getattr(cattle_out, "name", None),
            "buffalo": getattr(buffalo_out, "name", None),
        },
    }
    bundle = _make_bundle(
        engine="onnx", obj=session, path=path, spec=spec, meta=meta,
        cattle_map=maps[0], buffalo_map=maps[1], source=maps[2],
        num_cattle=num_cattle, num_buffalo=num_buffalo, pre=pre,
        device=providers[0],
    )
    bundle.onnx_input_name = input_meta.name
    bundle.onnx_output_names = [o.name for o in session.get_outputs()]
    return bundle


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def inspect_and_load(path: str, device: Any = "cpu", use_cuda: Optional[bool] = None
                     ) -> LoadedModelBundle:
    """Load ``.pt`` / ``.onnx`` with full introspection."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".onnx":
        if use_cuda is None:
            use_cuda = str(device).startswith("cuda") or torch.cuda.is_available()
        return inspect_and_load_onnx(path, use_cuda=use_cuda)
    return inspect_and_load_pt(path, device=device)


def describe_model_file(path: str) -> Dict[str, Any]:
    """Metadata + introspected spec without keeping the model in memory."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".onnx":
        bundle = inspect_and_load_onnx(path, use_cuda=False)
        return bundle.spec_summary()
    bundle = inspect_and_load_pt(path, device="cpu")
    return bundle.spec_summary()


# ---------------------------------------------------------------------------
# Metadata construction + embedding
# ---------------------------------------------------------------------------

def _read_trait_vocab() -> Dict[str, Any]:
    data = _read_json(TRAIT_FILE)
    return data if isinstance(data, dict) else {}


def build_model_metadata(*, state_dict: Optional[Dict[str, Any]] = None,
                         spec: Optional[Dict[str, Any]] = None,
                         backbone: Optional[str] = None,
                         attention: Optional[str] = None,
                         image_size: Optional[int] = None,
                         eval_resize: Optional[int] = None,
                         pad_to_square: Optional[bool] = None,
                         normalization_baked: bool = False,
                         input_dtype: str = "float32",
                         cattle_classes: Optional[Any] = None,
                         buffalo_classes: Optional[Any] = None,
                         search_dirs: Optional[Iterable[str]] = None,
                         split_dir: Optional[str] = None,
                         ood: Optional[Dict[str, Any]] = None,
                         trait_vocab: Optional[Dict[str, Any]] = None,
                         source_checkpoint: Optional[str] = None,
                         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the canonical metadata block for one model."""
    derived: Dict[str, Any] = dict(spec or {})
    if state_dict:
        derived.update({k: v for k, v in inspect_state_dict(state_dict).items()
                        if v is not None})

    num_cattle = derived.get("num_cattle")
    num_buffalo = derived.get("num_buffalo")
    dirs = list(search_dirs or [])
    if split_dir:
        dirs.append(split_dir)
    dirs.extend([SPLIT_DIR, PORTABLE_EXPORT_DIR])
    cattle, buffalo, source = resolve_class_maps(
        num_cattle, num_buffalo,
        embedded={"cattle_classes": cattle_classes, "buffalo_classes": buffalo_classes},
        search_dirs=dirs,
    )

    pre = default_preprocess({
        "image_size": image_size,
        "eval_resize": eval_resize,
        "pad_to_square": pad_to_square,
        "normalization_baked": normalization_baked,
        "input_dtype": input_dtype,
    })

    meta: Dict[str, Any] = {
        "model_meta_version": MODEL_META_VERSION,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "backbone": backbone or derived.get("backbone") or "lite2",
        "attention": attention or derived.get("attention") or "cbam",
        "feature_dim": int(derived.get("feature_dim") or 1280),
        "binary_dim": int(derived.get("binary_dim") or BINARY_DIM),
        "breed_dim": int(derived.get("breed_dim") or BREED_DIM),
        "cosine_head": bool(derived.get("cosine_head", False)),
        "has_trait_heads": bool(derived.get("has_trait_heads", False)),
        "num_cattle": int(num_cattle if num_cattle is not None else len(cattle)),
        "num_buffalo": int(num_buffalo if num_buffalo is not None else len(buffalo)),
        "image_size": int(pre["image_size"]),
        "eval_resize": int(pre["eval_resize"]),
        "pad_to_square": bool(pre["pad_to_square"]),
        "imagenet_mean": list(pre["imagenet_mean"]),
        "imagenet_std": list(pre["imagenet_std"]),
        "normalization_baked": bool(pre["normalization_baked"]),
        "normalization_mode": pre["normalization_mode"],
        "input_dtype": pre.get("input_dtype", input_dtype),
        "cattle_classes": cattle,
        "buffalo_classes": buffalo,
        "class_map_source": source,
        "ood": default_ood_config(ood),
    }
    vocab = trait_vocab if trait_vocab is not None else _read_trait_vocab()
    if vocab:
        meta["trait_vocab"] = vocab
    if source_checkpoint:
        meta["source_checkpoint"] = os.path.basename(source_checkpoint)
    if extra:
        meta.update(extra)
    return meta


def metadata_for_onnx(meta: Dict[str, Any]) -> Dict[str, str]:
    """Flatten metadata to ONNX ``StringStringEntryProto`` values."""
    flat: Dict[str, str] = {}
    for key, value in (meta or {}).items():
        if key == "trait_vocab":
            continue  # large; class maps + preprocessing are the contract
        if isinstance(value, str):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False)
    return flat


def _atomic_torch_save(obj: Any, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".meta_", suffix=".pt", dir=directory)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def embed_pt_metadata(path: str, meta: Optional[Dict[str, Any]] = None,
                      *, overwrite: bool = True, **meta_kwargs: Any) -> Dict[str, Any]:
    """Persist ``model_meta`` inside a ``.pt`` checkpoint or TorchScript file.

    Existing keys (epoch, metrics, optimizer state, ...) are preserved.  A raw
    ``state_dict`` is wrapped as ``{"state_dict": state, "model_meta": meta}``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".pt", ".pth"):
        raise ValueError(f"not a PyTorch model file: {path}")

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        ckpt = None

    if isinstance(ckpt, torch.nn.Module):
        return _embed_torchscript(path, meta, overwrite=overwrite, **meta_kwargs)

    if ckpt is None:
        # Likely a TorchScript archive that torch.load cannot unpickle.
        return _embed_torchscript(path, meta, overwrite=overwrite, **meta_kwargs)

    state = extract_state_dict(ckpt)
    existing = {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get(PT_META_KEY), dict):
        existing = ckpt[PT_META_KEY]
    if (existing.get("model_meta_version") and not overwrite):
        return existing

    if meta is None:
        meta = build_model_metadata(
            state_dict=state or None,
            source_checkpoint=path,
            ood=_ood_from_file(find_ood_json(path)),
            **meta_kwargs,
        )
    else:
        meta = copy.deepcopy(meta)

    if isinstance(ckpt, dict) and (("state_dict" in ckpt) or ("model_state_dict" in ckpt)):
        ckpt[PT_META_KEY] = meta
        _atomic_torch_save(ckpt, path)
    else:
        # Plain state_dict: wrap, but keep any non-tensor metadata we found.
        payload: Dict[str, Any] = {"state_dict": ckpt, PT_META_KEY: meta}
        _atomic_torch_save(payload, path)
    return meta


def _embed_torchscript(path: str, meta: Optional[Dict[str, Any]],
                       *, overwrite: bool = True, **meta_kwargs: Any) -> Dict[str, Any]:
    try:
        module = torch.jit.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"'{path}' is neither a checkpoint nor a TorchScript model: {exc}") from exc

    existing = _read_torchscript_metadata(path)
    if existing.get("model_meta_version") and not overwrite:
        return existing
    if meta is None:
        meta = build_model_metadata(
            spec=_infer_torchscript_spec(module),
            source_checkpoint=path,
            ood=_ood_from_file(find_ood_json(path)),
            **meta_kwargs,
        )
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".meta_", suffix=".pt", dir=directory)
    os.close(fd)
    try:
        module.save(tmp, _extra_files={
            TORCHSCRIPT_META_FILE: json.dumps(meta, ensure_ascii=False)})
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return meta


def embed_onnx_metadata(path: str, meta: Optional[Dict[str, Any]] = None,
                        *, overwrite: bool = True, **meta_kwargs: Any) -> Dict[str, Any]:
    """Persist metadata in an ONNX graph's ``metadata_props``.

    Duplicate keys are removed first so re-running the tool is idempotent.
    """
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("embedding ONNX metadata needs `pip install onnx`") from exc

    model = onnx.load(path, load_external_data=False)
    existing = {prop.key: _jsonish(prop.value) for prop in model.metadata_props}

    if meta is None:
        if existing.get("model_meta_version") and not overwrite:
            return existing
        # Infer architecture from the graph cheaply (no ORT session needed).
        spec: Dict[str, Any] = {}
        try:
            bundle = inspect_and_load_onnx(path, use_cuda=False)
            spec = dict(bundle.spec)
            spec["num_cattle"] = bundle.num_cattle
            spec["num_buffalo"] = bundle.num_buffalo
            existing = {**existing, **bundle.metadata}
            baked = bundle.normalization_baked
            image_size = bundle.image_size
        except Exception:
            spec["num_cattle"] = existing.get("num_cattle")
            spec["num_buffalo"] = existing.get("num_buffalo")
            baked = _detect_onnx_normalization(path, existing)
            image_size = existing.get("image_size")
        meta = build_model_metadata(
            spec=spec,
            image_size=image_size,
            normalization_baked=baked,
            source_checkpoint=path,
            ood=_ood_from_file(find_ood_json(path)),
            **meta_kwargs,
        )
    else:
        meta = copy.deepcopy(meta)

    # ONNX metadata is a string map: json-encode all non-strings.
    wanted = metadata_for_onnx(meta)
    kept = [prop for prop in model.metadata_props if prop.key not in wanted]
    del model.metadata_props[:]
    model.metadata_props.extend(kept)
    for key, value in wanted.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value

    if not model.doc_string:
        model.doc_string = "BreedID self-describing model (metadata in metadata_props)"

    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".meta_", suffix=".onnx", dir=directory)
    os.close(fd)
    try:
        onnx.save(model, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return meta


def embed_model_metadata(path: str, meta: Optional[Dict[str, Any]] = None,
                         *, overwrite: bool = True, **meta_kwargs: Any) -> Dict[str, Any]:
    """Dispatch to the PyTorch or ONNX embedder based on the extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".onnx":
        return embed_onnx_metadata(path, meta, overwrite=overwrite, **meta_kwargs)
    if ext in (".pt", ".pth"):
        return embed_pt_metadata(path, meta, overwrite=overwrite, **meta_kwargs)
    raise ValueError(f"unsupported model extension: {ext}")


def find_model_files(root: str) -> List[str]:
    """Every ``.pt`` / ``.pth`` / ``.onnx`` under ``root`` (sorted)."""
    found: List[str] = []
    for pattern in ("**/*.pt", "**/*.pth", "**/*.onnx"):
        found.extend(glob.glob(os.path.join(root, pattern), recursive=True))
    return sorted(set(found))


__all__ = [
    "MODEL_META_VERSION",
    "PT_META_KEY",
    "LoadedModelBundle",
    "sanitize_state_dict",
    "extract_state_dict",
    "inspect_state_dict",
    "resolve_class_maps",
    "default_preprocess",
    "default_ood_config",
    "find_ood_json",
    "build_transform",
    "read_pt_metadata",
    "read_onnx_metadata",
    "inspect_and_load_pt",
    "inspect_and_load_onnx",
    "inspect_and_load",
    "describe_model_file",
    "build_model_metadata",
    "metadata_for_onnx",
    "embed_pt_metadata",
    "embed_onnx_metadata",
    "embed_model_metadata",
    "find_model_files",
]
