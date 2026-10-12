"""Model definition: backbone, attention, classification heads, aux heads, OOD.

This file consolidates everything that defines or scores the network so the
model can be read top-to-bottom. It replaces the former ``efficientnet_lite.py``,
``cbam.py``, ``traits.py`` and ``ood_detector.py`` modules.

SECTION MAP
  1. Backbone (EfficientNet-Lite) ....... from former src/efficientnet_lite.py
  2. Attention (CBAM / SE) .............. from former src/cbam.py
  3. Cosine / ArcFace head .............. CosineHead (was in src/model.py)
  4. Classifier (BreedClassifier) ....... 3 heads + projection, from src/model.py
  5. Breed trait aux heads .............. from former src/traits.py (training-only)
  6. OOD detector ....................... from former src/ood_detector.py

REFERENCES
  - ``src/config.py`` is the single source of truth for every constant used here.
  - Trained by ``src/train.py``; exported by ``src/export.py``; evaluated by
    ``src/evaluate.py`` / ``src/parity_check.py``; sanity-checked by ``src/verify.py``.
  - ``test_model.py`` and ``server/model_manager.py`` import ``OODDetector`` here.
"""

# =======================================================================
# SECTION 4 — Classifier (BreedClassifier) + Cosine head
# BreedClassifier: backbone + attention + binary/cattle/buffalo heads, projection head (SupCon), predict()/predict_safe().
# Used by: src/train.py, src/export.py, src/evaluate.py, test_model.py
# Formerly src/model.py. Heads are built by _make_breed_head().
# =======================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (BINARY_DIM, BREED_DIM, CBAM_AFTER_STAGE, COSINE_SCALE,
                     DROPOUT, NUM_BUFFALO_BREEDS, NUM_CATTLE_BREEDS,
                     PROJECTION_DIM)


class CosineHead(nn.Module):
    """Normalised-feature scaled-cosine classifier (ArcFace-style).

    Forward returns ``scale * cos(theta)`` with NO additive margin, so inference
    and export are plain scaled-cosine logits. The angular margin is applied to
    the target class only during training, inside the loss (see
    ``train._apply_cosine_margin``), which keeps the exported graph margin-free.
    """

    def __init__(self, in_features, out_features, scale=COSINE_SCALE):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)

    def forward(self, x):
        cos = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
        return self.scale * cos


def _make_breed_head(feature_dim, num_classes, dropout, cosine=False,
                     scale=COSINE_SCALE, breed_dim=BREED_DIM):
    layers = [
        nn.Linear(feature_dim, breed_dim),
        nn.BatchNorm1d(breed_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(breed_dim, breed_dim // 2),
        nn.BatchNorm1d(breed_dim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
    ]
    layers.append(CosineHead(breed_dim // 2, num_classes, scale) if cosine
                  else nn.Linear(breed_dim // 2, num_classes))
    return nn.Sequential(*layers)


class BreedClassifier(nn.Module):
    def __init__(self, backbone="lite2", num_cattle=NUM_CATTLE_BREEDS,
                 num_buffalo=NUM_BUFFALO_BREEDS, cbam_stage=CBAM_AFTER_STAGE,
                 attention="cbam", activation="relu6", pretrained_path=None,
                 dropout=DROPOUT, cosine_head=False, cosine_scale=COSINE_SCALE,
                 binary_dim=BINARY_DIM, breed_dim=BREED_DIM):
        super().__init__()
        self.cosine_head = bool(cosine_head)
        self.cosine_scale = float(cosine_scale)
        self.backbone = EfficientNetLite(arch=backbone, activation=activation)
        self.cbam_stage = cbam_stage
        stage_channels = self.backbone.stage_channels[cbam_stage]
        self.attention = build_attention(attention, stage_channels)
        feature_dim = self.backbone.head_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.binary_head = nn.Sequential(
            nn.Linear(feature_dim, binary_dim),
            nn.BatchNorm1d(binary_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(binary_dim, binary_dim // 2),
            nn.BatchNorm1d(binary_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(binary_dim // 2, 2),
        )
        self.cattle_head = _make_breed_head(feature_dim, num_cattle, dropout,
                                            self.cosine_head, self.cosine_scale,
                                            breed_dim=breed_dim)
        self.buffalo_head = _make_breed_head(feature_dim, num_buffalo, dropout,
                                             self.cosine_head, self.cosine_scale,
                                             breed_dim=breed_dim)

        # Auxiliary projection head used only by the supervised-contrastive
        # loss (never exported). The 1280-d pooled features were previously
        # returned but unused; this gives the model an explicit objective to
        # cluster embeddings by breed, which is what separates near-identical
        # indigenous breeds.
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, PROJECTION_DIM),
        )

        if pretrained_path:
            n = load_backbone_weights(self.backbone, pretrained_path)
            print(f"[model] loaded {n} tensors from {pretrained_path}")

    def forward_features(self, x):
        feats = self.backbone.forward_until(x, self.cbam_stage)
        feats = self.attention(feats)
        feats = self.backbone.forward_from(feats, self.cbam_stage + 1)
        return self.avg_pool(feats).flatten(1)

    def forward(self, x):
        pooled = self.forward_features(x)
        return {
            "binary": self.binary_head(pooled),
            "cattle": self.cattle_head(pooled),
            "buffalo": self.buffalo_head(pooled),
            "features": pooled,
            "embedding": self.projection_head(pooled),
        }

    @torch.no_grad()
    def predict(self, x):
        """Soft species routing: never let a hard binary argmax discard the
        other head. Returns the combined 75-class breed distribution, built as
        p(species) * softmax(breed head for that species).
        """
        out = self.forward(x)
        p_species = F.softmax(out["binary"], dim=1)
        p_cattle = F.softmax(out["cattle"], dim=1)
        p_buffalo = F.softmax(out["buffalo"], dim=1)
        return torch.cat([p_species[:, 0:1] * p_cattle,
                          p_species[:, 1:2] * p_buffalo], dim=1)

    @torch.no_grad()
    def predict_safe(self, x, ood_detector=None):
        """Like predict(), but returns zero-confidence result for OOD inputs."""
        out = self.forward(x)
        
        if ood_detector is not None:
            is_ood, score, details = ood_detector.is_ood(
                out["binary"], out["cattle"], out["buffalo"]
            )
            if is_ood:
                # Return zero distribution — "not a recognized animal"
                n_total = out["cattle"].shape[1] + out["buffalo"].shape[1]
                return torch.zeros(x.shape[0], n_total, device=x.device), {
                    "ood": True, "ood_score": score, **details
                }
        
        # Normal soft routing
        p_species = F.softmax(out["binary"], dim=1)
        p_cattle = F.softmax(out["cattle"], dim=1)
        p_buffalo = F.softmax(out["buffalo"], dim=1)
        combined = torch.cat([p_species[:, 0:1] * p_cattle,
                              p_species[:, 1:2] * p_buffalo], dim=1)
        return combined, {"ood": False}

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.attention.parameters():
            p.requires_grad = False

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def backbone_eval(self):
        self.backbone.eval()
        self.attention.eval()

    def backbone_train(self):
        self.backbone.train()
        self.attention.train()

# =======================================================================
# SECTION 1 — Backbone (EfficientNet-Lite)
# Defines EfficientNetLite (stem + MBConv stages) and load_backbone_weights().
# Used by: BreedClassifier, src/verify.py
# Formerly src/efficientnet_lite.py.
# =======================================================================
import torch
import torch.nn as nn

LITE_ARCHS = {
    "lite2": {
        "stem_channels": 32,
        "stages": [
            {"repeat": 1, "kernel": 3, "stride": 1, "expand": 1, "in": 32,  "out": 16},
            {"repeat": 3, "kernel": 3, "stride": 2, "expand": 6, "in": 16,  "out": 24},
            {"repeat": 3, "kernel": 5, "stride": 2, "expand": 6, "in": 24,  "out": 48},
            {"repeat": 4, "kernel": 3, "stride": 2, "expand": 6, "in": 48,  "out": 88},
            {"repeat": 4, "kernel": 5, "stride": 2, "expand": 6, "in": 88,  "out": 120},
            {"repeat": 5, "kernel": 5, "stride": 2, "expand": 6, "in": 120, "out": 208},
            {"repeat": 1, "kernel": 3, "stride": 1, "expand": 6, "in": 208, "out": 352},
        ],
        "head_channels": 1280,
    },
    "lite4": {
        "stem_channels": 32,
        "stages": [
            {"repeat": 1, "kernel": 3, "stride": 1, "expand": 1, "in": 32,  "out": 24},
            {"repeat": 4, "kernel": 3, "stride": 2, "expand": 6, "in": 24,  "out": 32},
            {"repeat": 4, "kernel": 5, "stride": 2, "expand": 6, "in": 32,  "out": 56},
            {"repeat": 6, "kernel": 3, "stride": 2, "expand": 6, "in": 56,  "out": 112},
            {"repeat": 6, "kernel": 5, "stride": 2, "expand": 6, "in": 112, "out": 160},
            {"repeat": 8, "kernel": 5, "stride": 2, "expand": 6, "in": 160, "out": 272},
            {"repeat": 1, "kernel": 3, "stride": 1, "expand": 6, "in": 272, "out": 448},
        ],
        "head_channels": 1280,
    },
}


def make_activation(name="relu6"):
    name = (name or "relu6").lower()
    if name in ("relu6", "relu6_inplace"):
        return nn.ReLU6(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"Unknown activation {name}")


class MBConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel, stride, expand, activation):
        super().__init__()
        self.activation = activation
        self._add = stride == 1 and cin == cout
        expanded = cin * expand
        self._depthwise_conv = nn.Conv2d(
            expanded, expanded, kernel, stride=stride, padding=kernel // 2,
            groups=expanded, bias=False)
        self._bn1 = nn.BatchNorm2d(expanded)
        if expand != 1:
            self._expand_conv = nn.Conv2d(cin, expanded, 1, bias=False)
            self._bn0 = nn.BatchNorm2d(expanded)
        self._project_conv = nn.Conv2d(expanded, cout, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        if self._add:
            residual = x
        if hasattr(self, "_expand_conv"):
            x = self.activation(self._bn0(self._expand_conv(x)))
        x = self.activation(self._bn1(self._depthwise_conv(x)))
        x = self._bn2(self._project_conv(x))
        if self._add:
            x = x + residual
        return x


class EfficientNetLite(nn.Module):
    def __init__(self, arch="lite2", activation="relu6"):
        super().__init__()
        spec = LITE_ARCHS[arch]
        self.arch = arch
        self.head_channels = spec["head_channels"]
        self.stage_channels = [s["out"] for s in spec["stages"]]
        act = make_activation(activation)

        self.stem = nn.Sequential(
            nn.Conv2d(3, spec["stem_channels"], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(spec["stem_channels"]),
            act,
        )
        self.blocks = nn.ModuleList()
        for stage in spec["stages"]:
            stage_modules = nn.ModuleList()
            for rep in range(stage["repeat"]):
                stage_modules.append(MBConvBlock(
                    cin=stage["in"] if rep == 0 else stage["out"],
                    cout=stage["out"],
                    kernel=stage["kernel"],
                    stride=stage["stride"] if rep == 0 else 1,
                    expand=stage["expand"],
                    activation=act,
                ))
            self.blocks.append(stage_modules)
        self.head = nn.Sequential(
            nn.Conv2d(spec["stages"][-1]["out"], spec["head_channels"], 1, bias=False),
            nn.BatchNorm2d(spec["head_channels"]),
            act,
        )

    def forward_until(self, x, stage_index):
        x = self.stem(x)
        for i in range(stage_index + 1):
            for block in self.blocks[i]:
                x = block(x)
        return x

    def forward_from(self, x, stage_index):
        for i in range(stage_index, len(self.blocks)):
            for block in self.blocks[i]:
                x = block(x)
        x = self.head(x)
        return x

    def forward(self, x):
        last = len(self.blocks) - 1
        return self.forward_from(self.forward_until(x, last), last + 1)


def load_backbone_weights(backbone, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint contents in {path}: {type(checkpoint)}")
    expected = set(backbone.state_dict().keys())
    provided = set(checkpoint.keys())
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    allowed_missing = set()
    allowed_unexpected = {"fc.weight", "fc.bias"}
    hard_missing = [k for k in missing if k not in allowed_missing]
    hard_unexpected = [k for k in unexpected if k not in allowed_unexpected]
    if hard_missing or hard_unexpected:
        raise ValueError(
            f"Weight mismatch for {path}:\n"
            f"  missing: {hard_missing[:8]}\n"
            f"  unexpected: {hard_unexpected[:8]}")
    backbone.load_state_dict(checkpoint, strict=False)
    return len(provided)

# =======================================================================
# SECTION 2 — Attention (CBAM / SE)
# ChannelAttention + SpatialAttention (CBAM), SEBlock, build_attention().
# Used by: BreedClassifier.attention
# Formerly src/cbam.py.
# =======================================================================
import torch
import torch.nn as nn

from .config import CBAM_IDENTITY_INIT


def _zero_init_last(seq):
    """Zero the last Conv2d weight in a Sequential so its output is 0."""
    for layer in reversed(list(seq)):
        if isinstance(layer, nn.Conv2d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.identity_init = identity_init
        if identity_init:
            _zero_init_last(self.mlp)

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        att = self.mlp(avg) + self.mlp(mx)
        if self.identity_init:
            # gate is 0 at init (zero-init weights), so x*(1+gate) == x.
            gate = 2.0 * self.sigmoid(att) - 1.0
            return x * (1.0 + gate)
        return x * self.sigmoid(att)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.identity_init = identity_init
        if identity_init:
            nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        fused = torch.cat([avg, mx], dim=1)
        att = self.conv(fused)
        if self.identity_init:
            gate = 2.0 * self.sigmoid(att) - 1.0
            return x * (1.0 + gate)
        return x * self.sigmoid(att)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7,
                 identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction,
                                                  identity_init=identity_init)
        self.spatial_attention = SpatialAttention(kernel_size,
                                                  identity_init=identity_init)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.identity_init = identity_init
        if identity_init:
            _zero_init_last(self.fc)

    def forward(self, x):
        pooled = self.pool(x)
        att = self.fc[:-1](pooled) if self.identity_init else self.fc(pooled)
        if self.identity_init:
            gate = 2.0 * torch.sigmoid(att) - 1.0
            return x * (1.0 + gate)
        return x * self.fc(pooled)


def build_attention(kind, channels):
    kind = (kind or "cbam").lower()
    if kind == "cbam":
        return CBAM(channels)
    if kind == "se":
        return SEBlock(channels)
    raise ValueError(f"Unknown attention kind {kind}")

# =======================================================================
# SECTION 5 — Breed trait auxiliary heads
# TraitClassifier + vocab/target builders + evaluate_traits().
# Used by: src/train.py (--trait-weight)
# Training-only; excluded from export. Formerly src/traits.py.
# Values: data/breed_traits.json.
# =======================================================================
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_BUFFALO_BREEDS, NUM_CATTLE_BREEDS


def load_trait_spec(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_trait_vocab(spec):
    """{field: {value: index}} for fields that have at least one filled value."""
    if not spec:
        return {}
    vocab = {}
    for field in spec.get("fields", []):
        values = set()
        for species in ("cattle", "buffalo"):
            for _breed, tv in spec.get("breeds", {}).get(species, {}).items():
                v = (tv.get(field) or "").strip()
                if v:
                    values.add(v)
        if values:
            vocab[field] = {v: i for i, v in enumerate(sorted(values))}
    return vocab


def breed_trait_targets(spec, vocab, cattle_classes, buffalo_classes):
    """Per-field LongTensor[75]: global class id -> trait index, or -1 to ignore.

    Global ids follow ``_combined_class_ids``: cattle 0..C-1, buffalo
    C..C+B-1, where C/B are the SURVIVING class counts (dynamic head sizing).
    """
    n = len(cattle_classes) + len(buffalo_classes)
    targets = {f: torch.full((n,), -1, dtype=torch.long) for f in vocab}
    for species, cmap, offset in (("cattle", cattle_classes, 0),
                                  ("buffalo", buffalo_classes, len(cattle_classes))):
        for breed, idx in cmap.items():
            tv = spec.get("breeds", {}).get(species, {}).get(breed, {})
            for field in vocab:
                val = (tv.get(field) or "").strip()
                if val in vocab[field]:
                    targets[field][offset + idx] = vocab[field][val]
    return targets


class TraitClassifier(nn.Module):
    def __init__(self, in_dim, vocab, hidden=256, dropout=0.2):
        super().__init__()
        self.vocab = vocab
        self.fields = list(vocab.keys())
        self.heads = nn.ModuleDict({
            field: nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, len(vocab[field])),
            )
            for field in self.fields
        })

    def forward(self, features):
        return {field: self.heads[field](features) for field in self.fields}

    def loss(self, features, class_ids, targets, label_smoothing=0.0):
        """Masked multi-field CE. Returns (mean_loss, {field: acc})."""
        logits = self.forward(features)
        total = features.new_zeros(())
        n_fields = 0
        accs = {}
        for field in self.fields:
            tgt = targets[field].to(class_ids.device)[class_ids]
            valid = tgt >= 0
            if not valid.any():
                continue
            total = total + F.cross_entropy(logits[field][valid], tgt[valid],
                                            label_smoothing=label_smoothing)
            n_fields += 1
            accs[field] = (logits[field][valid].argmax(1) == tgt[valid]).float().mean().item()
        if n_fields:
            total = total / n_fields
        return total, accs


@torch.no_grad()
def evaluate_traits(model, trait_module, loader, targets, device,
                    max_batches=None):
    """Per-field accuracy over a loader (soft/hard-agnostic; uses features)."""
    from .train import _combined_class_ids
    trait_module.eval()
    totals = {f: [0, 0] for f in trait_module.fields}
    for step, (images, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        feats = model(images)["features"]
        ids = _combined_class_ids(labels)
        logits = trait_module(feats)
        for field in trait_module.fields:
            tgt = targets[field].to(device)[ids]
            valid = tgt >= 0
            if valid.any():
                totals[field][0] += (logits[field][valid].argmax(1)
                                     == tgt[valid]).sum().item()
                totals[field][1] += int(valid.sum())
    return {f: (c / n if n else None) for f, (c, n) in totals.items()}

# =======================================================================
# =======================================================================
# SECTION 6 — OOD detector
# OODDetector: post-hoc energy + MSP out-of-distribution scoring on logits.
# Used by: test_model.py, server/model_manager.py, scripts/calibrate_ood.py, src/train.py
# Consolidated from former src/ood_detector.py.
# =======================================================================
import datetime
import glob
import json
import os
import re
import time

import torch
import torch.nn.functional as F

from .config import (
    OOD_DIR,
    OOD_ENERGY_TEMPERATURE,
    OOD_ENERGY_THRESHOLD,
    OOD_MSP_THRESHOLD,
    OOD_BINARY_CONFIDENCE_FLOOR,
    OOD_PERCENTILE,
    PORTABLE_EXPORT_DIR,
    PROJECT_ROOT,
)


class OODDetector:
    """Post-hoc energy-based out-of-distribution detector.

    Computes an OOD score from raw model logits without requiring retraining.
    Works with both PyTorch and ONNX models (torch tensors and numpy arrays).

    Supports:
      1. Per-model calibrated thresholds (e.g. outputs/ood/<model_name>.json).
      2. Custom selected OOD configuration files.
      3. Global fallback to outputs/ood_thresholds.json or config defaults.
      4. Auto-calibration after model training.
    """

    DEFAULT_CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "outputs", "ood_thresholds.json")

    def __init__(self, model_name=None, config_path=None,
                 temperature=OOD_ENERGY_TEMPERATURE,
                 energy_threshold=OOD_ENERGY_THRESHOLD,
                 msp_threshold=OOD_MSP_THRESHOLD,
                 binary_confidence_floor=OOD_BINARY_CONFIDENCE_FLOOR):
        self.model_name = model_name
        self.source_file = None
        self.profile_name = "default"
        self.metadata = {}

        # 1. Resolve configuration file if custom path or model name is given
        resolved_path = None
        if config_path and os.path.exists(config_path):
            resolved_path = config_path
        elif model_name:
            resolved_path = self.resolve_profile_path(model_name)

        if resolved_path is None and os.path.exists(self.DEFAULT_CALIBRATION_FILE):
            resolved_path = self.DEFAULT_CALIBRATION_FILE

        if resolved_path and os.path.exists(resolved_path):
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    cal = json.load(f)
                temperature = cal.get("energy_temperature", temperature)
                energy_threshold = cal.get("energy_threshold", energy_threshold)
                msp_threshold = cal.get("msp_threshold", msp_threshold)
                binary_confidence_floor = cal.get("binary_confidence_floor",
                                                   binary_confidence_floor)
                self.source_file = resolved_path
                self.profile_name = os.path.basename(resolved_path)
                self.metadata = cal
            except Exception:
                pass

        self.T = float(temperature)
        self.energy_threshold = float(energy_threshold)
        self.msp_threshold = float(msp_threshold)
        self.binary_confidence_floor = float(binary_confidence_floor)

    @classmethod
    def clean_name(cls, name: str) -> str:
        """Strip directory prefixes, file extensions, and UI tags from model name."""
        if not name:
            return "model"
        base = os.path.basename(name)
        base = re.sub(r"^\[.*?\]\s*", "", base)
        base = re.sub(r"\s*\(ONNX\)$", "", base, flags=re.IGNORECASE)
        for ext in (".pt", ".onnx", ".json"):
            if base.lower().endswith(ext):
                base = base[:-len(ext)]
        base = re.sub(r"[^\w\-_.]", "_", base)
        return base.strip()

    @classmethod
    def resolve_profile_path(cls, model_name: str):
        """Find the matching OOD calibration JSON for a given model."""
        if not model_name:
            return None

        if os.path.isabs(model_name) and os.path.exists(model_name):
            return model_name

        clean = cls.clean_name(model_name)

        candidate = os.path.join(OOD_DIR, f"{clean}.json")
        if os.path.exists(candidate):
            return candidate

        candidate_ood = os.path.join(OOD_DIR, f"{clean}_ood.json")
        if os.path.exists(candidate_ood):
            return candidate_ood

        portable_dirs = glob.glob(os.path.join(PORTABLE_EXPORT_DIR, "*"))
        for pdir in portable_dirs:
            if clean in os.path.basename(pdir):
                p_ood = os.path.join(pdir, "ood_thresholds.json")
                if os.path.exists(p_ood):
                    return p_ood

        if os.path.exists(OOD_DIR):
            for fname in os.listdir(OOD_DIR):
                if fname.endswith(".json"):
                    stem = fname[:-5]
                    if stem in clean or clean in stem:
                        return os.path.join(OOD_DIR, fname)

        if os.path.exists(cls.DEFAULT_CALIBRATION_FILE):
            return cls.DEFAULT_CALIBRATION_FILE

        return None

    @classmethod
    def calibrate(cls, model, loader, device, percentile=OOD_PERCENTILE,
                  temperature=OOD_ENERGY_TEMPERATURE,
                  binary_floor=OOD_BINARY_CONFIDENCE_FLOOR) -> dict:
        """Calibrate energy and MSP thresholds on an in-distribution DataLoader."""
        device = torch.device(device) if isinstance(device, str) else device
        model.eval()

        detector = cls(temperature=temperature,
                       binary_confidence_floor=binary_floor)
        energies = []
        msps = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device)

                with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    out = model(images)

                b_logits = out["binary"]
                c_logits = out["cattle"]
                buf_logits = out["buffalo"]

                e_batch = detector.compute_energy(b_logits, c_logits, buf_logits)
                m_batch = detector.compute_msp(b_logits)

                energies.extend(e_batch.detach().cpu().tolist())
                msps.extend(m_batch.detach().cpu().tolist())

        if not energies:
            raise ValueError("[OODDetector] Calibration failed: loader yielded 0 samples.")

        energies_t = torch.tensor(energies, dtype=torch.float32)
        msps_t = torch.tensor(msps, dtype=torch.float32)

        energy_threshold = torch.quantile(energies_t, percentile / 100.0).item()
        msp_threshold = torch.quantile(msps_t, (100.0 - percentile) / 100.0).item()

        return {
            "energy_temperature": float(temperature),
            "energy_threshold": round(float(energy_threshold), 4),
            "msp_threshold": round(float(msp_threshold), 4),
            "binary_confidence_floor": float(binary_floor),
            "percentile": float(percentile),
            "calibrated_on_samples": len(energies),
            "calibrated_at": datetime.datetime.now().isoformat(),
            "energy_stats": {
                "mean": round(float(energies_t.mean().item()), 4),
                "std": round(float(energies_t.std().item()), 4),
                "min": round(float(energies_t.min().item()), 4),
                "max": round(float(energies_t.max().item()), 4),
            },
            "msp_stats": {
                "mean": round(float(msps_t.mean().item()), 4),
                "std": round(float(msps_t.std().item()), 4),
                "min": round(float(msps_t.min().item()), 4),
                "max": round(float(msps_t.max().item()), 4),
            }
        }

    @classmethod
    def save(cls, thresholds: dict, name: str, out_dir: str = None) -> str:
        """Save calibrated thresholds to JSON with appropriate naming."""
        clean = cls.clean_name(name)
        target_dir = out_dir if out_dir is not None else OOD_DIR
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, f"{clean}.json")
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)

        # Update global fallback outputs/ood_thresholds.json
        try:
            with open(cls.DEFAULT_CALIBRATION_FILE, "w", encoding="utf-8") as f:
                json.dump(thresholds, f, indent=2)
        except Exception:
            pass

        return target_path

    @classmethod
    def list_available_profiles(cls) -> list:
        """Enumerate all available OOD calibration files and their metrics."""
        profiles = []
        seen_paths = set()

        def _add_profile(path, label=None, is_default=False):
            if path in seen_paths or not os.path.exists(path):
                return
            seen_paths.add(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                fname = label or os.path.basename(path)
                profiles.append({
                    "id": fname,
                    "filename": fname,
                    "path": path,
                    "is_default": is_default,
                    "energy_threshold": data.get("energy_threshold", OOD_ENERGY_THRESHOLD),
                    "msp_threshold": data.get("msp_threshold", OOD_MSP_THRESHOLD),
                    "binary_confidence_floor": data.get("binary_confidence_floor", OOD_BINARY_CONFIDENCE_FLOOR),
                    "energy_temperature": data.get("energy_temperature", OOD_ENERGY_TEMPERATURE),
                    "percentile": data.get("percentile", OOD_PERCENTILE),
                    "calibrated_on_samples": data.get("calibrated_on_samples"),
                    "calibrated_at": data.get("calibrated_at"),
                })
            except Exception:
                pass

        # 1. Models in OOD_DIR
        if os.path.exists(OOD_DIR):
            for f in sorted(os.listdir(OOD_DIR)):
                if f.endswith(".json"):
                    _add_profile(os.path.join(OOD_DIR, f))

        # 2. Portable export folders
        for pdir in sorted(glob.glob(os.path.join(PORTABLE_EXPORT_DIR, "*"))):
            p_ood = os.path.join(pdir, "ood_thresholds.json")
            if os.path.exists(p_ood):
                _add_profile(p_ood, label=f"portable/{os.path.basename(pdir)}")

        # 3. Default fallback
        if os.path.exists(cls.DEFAULT_CALIBRATION_FILE):
            _add_profile(cls.DEFAULT_CALIBRATION_FILE, label="ood_thresholds.json", is_default=True)

        return profiles

    def compute_energy(self, *logits_list):
        """E(x) = -T * log(Σ exp(f_i / T)) across all logit heads."""
        is_numpy = not isinstance(logits_list[0], torch.Tensor)
        if is_numpy:
            import numpy as np
            combined_logits = np.concatenate(logits_list, axis=-1)
            max_val = np.max(combined_logits / self.T, axis=-1, keepdims=True)
            sum_exp = np.sum(np.exp(combined_logits / self.T - max_val), axis=-1)
            energy = -self.T * (np.squeeze(max_val, axis=-1) + np.log(sum_exp))
            return energy
        else:
            combined_logits = torch.cat(logits_list, dim=-1)
            energy = -self.T * torch.logsumexp(combined_logits / self.T, dim=-1)
            return energy

    def compute_msp(self, binary_logits):
        """Max softmax probability from binary head (cattle-vs-buffalo)."""
        is_numpy = not isinstance(binary_logits, torch.Tensor)
        if is_numpy:
            import numpy as np
            exp_logits = np.exp(binary_logits - np.max(binary_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return np.max(probs, axis=-1)
        else:
            probs = F.softmax(binary_logits, dim=-1)
            return torch.max(probs, dim=-1)[0]

    def is_ood(self, binary_logits, cattle_logits, buffalo_logits):
        """Returns (is_ood: bool, ood_score: float, details: dict).

        A sample is considered OOD if:
          1. Its energy is strictly greater than the energy threshold, OR
          2. Its binary maximum softmax probability (MSP) is lower than the
             MSP threshold, OR
          3. Its binary confidence (max of P(cattle), P(buffalo)) is below
             the binary confidence floor.
        """
        energy = self.compute_energy(binary_logits, cattle_logits, buffalo_logits)
        msp = self.compute_msp(binary_logits)

        is_numpy = not isinstance(energy, torch.Tensor)
        if is_numpy:
            import numpy as np
            energy_val = float(np.atleast_1d(energy)[0])
            msp_val = float(np.atleast_1d(msp)[0])
        else:
            energy_val = energy.item() if energy.numel() == 1 else energy[0].item()
            msp_val = msp.item() if msp.numel() == 1 else msp[0].item()

        ood_by_energy = energy_val > self.energy_threshold
        ood_by_msp = msp_val < self.msp_threshold
        ood_by_binary_floor = msp_val < self.binary_confidence_floor
        is_ood_flag = ood_by_energy or ood_by_msp or ood_by_binary_floor

        return is_ood_flag, energy_val, {
            "energy": energy_val,
            "energy_threshold": self.energy_threshold,
            "msp": msp_val,
            "msp_threshold": self.msp_threshold,
            "binary_confidence_floor": self.binary_confidence_floor,
            "ood_by_energy": ood_by_energy,
            "ood_by_msp": ood_by_msp,
            "ood_by_binary_floor": ood_by_binary_floor,
            "source_file": self.source_file,
            "profile_name": self.profile_name,
        }

