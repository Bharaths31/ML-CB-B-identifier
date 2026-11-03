"""Post-hoc gender & age estimation (+ trait label decoding) for inference.

The breed checkpoint was trained for species/breed only. Everything here is
additive and runs *after* ``BreedClassifier`` -- no retraining, no change to the
exported graph, and the OOD gate is honoured (attributes are skipped for
out-of-distribution inputs).

TWO ENGINES
  1. CLIP zero-shot (primary) ............ ``openai/clip-vit-base-patch32``
     Prompt-conditioned on the predicted species. Lazy-loaded inside a
     try/except so an offline install, a missing ``transformers`` package or a
     failed download degrades to the heuristic instead of crashing a request.
  2. Morphological heuristic (fallback) .. trait-head activations (hump /
     horn / dewlap / size), subject aspect ratio and dairy-breed priors.
     Always available; explicitly marked ``morphological_heuristic`` and
     flagged for one-click farmer confirmation because it is a prior, not a
     measurement.

Trait *labels* also live here: ``decode_traits`` turns the raw argmax indices
from the attached ``model.trait_heads`` into the human-readable strings the API
returns (Hump: "Large / Prominent", ...).

USED BY
  - ``test_model.py`` (GUI + its ``/api/predict``)
  - ``server/services/inference.py`` (Flask ``/api/predict``)
"""

import threading

import numpy as np
import torch
import torch.nn.functional as F

from .config import (ATTRIBUTES_CLIP_CACHE_DIR, ATTRIBUTES_CLIP_ENABLED,
                     ATTRIBUTES_CLIP_MODEL_ID, ATTRIBUTES_CLIP_TEMPERATURE,
                     ATTRIBUTES_ENABLED, ATTRIBUTES_FAST_MODE, TRAIT_FIELDS,
                     ATTRIBUTES_CLIP_ALLOW_DOWNLOAD)
from .trait_labels import (TRAIT_DISPLAY, TRAIT_GENERIC_LABELS,  # re-exported
                           humanize_trait)

# ---------------------------------------------------------------------------
#  Labels & human-readable trait vocabulary
# ---------------------------------------------------------------------------

GENDER_LABELS = {
    "cattle": {"female": "Female (Cow)", "male": "Male (Bull / Ox)"},
    "buffalo": {"female": "Female (Buffalo)", "male": "Male (Buffalo Bull)"},
}
GENDER_SHORT = {"female": "Female", "male": "Male"}

AGE_LABELS = {
    "calf": "Calf (< 1 yr)",
    "young": "Young / Sub-Adult (1 - 3 yrs)",
    "adult": "Adult (> 3 yrs)",
}
AGE_SHORT = {"calf": "Calf", "young": "Young", "adult": "Adult"}

GENDER_NOTE_CLIP = "Visual estimation from body/udder profile"
AGE_NOTE_CLIP = "Visual stage estimate; verify via dentition (teeth) for exact age"
GENDER_NOTE_HEURISTIC = ("Morphology + breed prior only (no udder segmentation); "
                         "confirm with the farmer")
AGE_NOTE_HEURISTIC = ("Body-size/horn-development prior; confirm with the farmer "
                      "or dentition")

# Breeds where commercial datasets overwhelmingly photograph productive
# females (~85% of dairy imagery). Used only as a soft prior.
DAIRY_BREEDS = {
    "holstein_friesian", "holstein", "jersey", "ayrshire", "sahiwal",
    "red_sindhi", "gir", "tharparkar", "kankrej", "rathi", "mehsana",
    "murrah", "surti", "nili_ravi", "jaffarabadi", "bhadawari", "nagpuri",
    "pandharpuri", "chilika",
}


def humanize_trait(field, token):
    """Map a snake_case vocabulary token to a display string."""
    if token is None:
        return "Unknown"
    table = TRAIT_DISPLAY.get(field, {})
    if token in table:
        return table[token]
    generic = TRAIT_GENERIC_LABELS.get(field, [])
    if token.startswith("class_"):
        try:
            idx = int(token.split("_", 1)[1])
        except (IndexError, ValueError):
            return token
        if 0 <= idx < len(generic):
            return generic[idx]
        return f"{field.title()} class {idx}"
    return token.replace("_", " ").title()


def decode_traits(trait_logits, vocab):
    """Turn ``{field: logits}`` into display labels + per-field detail.

    Returns ``(labels, details)``:
      labels  -> ``{"hump": "Large / Prominent", ...}`` (API-ready)
      details -> index/token/confidence/top-3, for developer views & heuristics
    """
    labels, details = {}, {}
    for field in TRAIT_FIELDS:
        logits = trait_logits.get(field)
        if logits is None:
            continue
        logits = logits.detach().float()
        if logits.dim() > 1:          # [1, C] from the batched trait heads
            logits = logits[0]
        probs = F.softmax(logits, dim=0)
        idx = int(torch.argmax(probs).item())
        inv = {i: token for token, i in (vocab.get(field) or {}).items()}
        token = inv.get(idx, f"class_{idx}")
        labels[field] = humanize_trait(field, token)

        k = min(3, probs.numel())
        top_p, top_i = torch.topk(probs, k)
        details[field] = {
            "index": idx,
            "token": token,
            "label": labels[field],
            "confidence": round(float(probs[idx].item()) * 100, 2),
            "top3": [
                {
                    "index": int(i),
                    "token": inv.get(int(i), f"class_{int(i)}"),
                    "label": humanize_trait(field, inv.get(int(i), f"class_{int(i)}")),
                    "confidence": round(float(p) * 100, 2),
                }
                for p, i in zip(top_p.tolist(), top_i.tolist())
            ],
        }
    return labels, details


# ---------------------------------------------------------------------------
#  Prompt sets (exactly as specified; {species} is lower-cased)
# ---------------------------------------------------------------------------

GENDER_PROMPTS = {
    "female": ("a side profile photo of a female dairy {species} cow with an "
               "udder, slender neck, and feminine face"),
    "male": ("a side profile photo of a muscular male {species} bull or ox with "
             "a heavy neck, large hump, and broad head"),
}

AGE_PROMPTS = {
    "calf": ("a photo of a small young baby calf with short legs, small head, "
             "and no horn development"),
    "young": ("a photo of a medium-sized young heifer or sub-adult {species} "
              "with developing horns and lean body"),
    "adult": ("a photo of a fully grown mature adult {species} with full body "
              "mass and developed horns"),
}


def _softmax(scores):
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr - arr.max()
    exp = np.exp(arr)
    return exp / exp.sum()


def _species_key(species):
    key = str(species or "").strip().lower()
    return "buffalo" if key.startswith("buff") else "cattle"


def _breed_key(breed):
    key = str(breed or "").strip().lower()
    for suffix in (" cattle", " buffalo"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key.replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
#  Image subject statistics (cheap, deterministic, no extra model)
# ---------------------------------------------------------------------------

def subject_stats(image, sample_size=96):
    """Estimate the animal's foreground box from border-colour similarity.

    Deliberately simple: downsample, treat the median border colour as
    background, threshold the colour distance, and take the resulting bounding
    box. Used only as a weak cue for "compact/upright" vs "long side profile",
    so it degrades gracefully to the full frame when segmentation fails.
    """
    try:
        small = image.convert("RGB")
        small.thumbnail((sample_size, sample_size))
        arr = np.asarray(small, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[0] < 4 or arr.shape[1] < 4:
            raise ValueError("image too small")

        border = np.concatenate([
            arr[:2].reshape(-1, 3), arr[-2:].reshape(-1, 3),
            arr[:, :2].reshape(-1, 3), arr[:, -2:].reshape(-1, 3),
        ])
        background = np.median(border, axis=0)
        dist = np.linalg.norm(arr - background, axis=2)
        threshold = max(18.0, float(np.percentile(dist, 60.0)))
        mask = dist > threshold
        coverage = float(mask.mean())

        if coverage < 0.02:
            raise ValueError("no foreground")
        ys, xs = np.where(mask)
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        return {
            "source": "foreground",
            "aspect_ratio": round(width / max(1, height), 3),
            "coverage": round(coverage, 3),
            "box_px": [int(xs.min()), int(ys.min()), width, height],
            "sample_grid": [int(arr.shape[1]), int(arr.shape[0])],
            "image_aspect": round(image.width / max(1, image.height), 3),
        }
    except Exception:
        return {
            "source": "full_frame",
            "aspect_ratio": round(image.width / max(1, image.height), 3),
            "coverage": 1.0,
            "box_px": [0, 0, image.width, image.height],
            "image_aspect": round(image.width / max(1, image.height), 3),
        }


# ---------------------------------------------------------------------------
#  Estimator
# ---------------------------------------------------------------------------

class AttributeEstimator:
    """Gender / age estimator with a CLIP engine and an offline fallback."""

    def __init__(self, device=None, clip_enabled=None, fast_mode=None,
                 model_id=ATTRIBUTES_CLIP_MODEL_ID,
                 temperature=ATTRIBUTES_CLIP_TEMPERATURE,
                 cache_dir=ATTRIBUTES_CLIP_CACHE_DIR):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.clip_enabled = ATTRIBUTES_CLIP_ENABLED if clip_enabled is None else bool(clip_enabled)
        self.fast_mode = ATTRIBUTES_FAST_MODE if fast_mode is None else bool(fast_mode)
        self.model_id = model_id
        self.temperature = max(1e-3, float(temperature))
        self.cache_dir = cache_dir

        self._clip_state = "unloaded"   # unloaded | loading | ready | unavailable
        self._clip_error = None
        self._clip_model = None
        self._clip_processor = None
        self._lock = threading.Lock()

    # -- CLIP plumbing ---------------------------------------------------
    @property
    def clip_status(self):
        return {"state": self._clip_state, "model": self.model_id,
                "error": self._clip_error}

    def configure(self, clip_enabled=None, fast_mode=None):
        """Apply runtime toggles (e.g. the UI "Fast Mode" switch)."""
        if clip_enabled is not None:
            self.clip_enabled = bool(clip_enabled)
        if fast_mode is not None:
            self.fast_mode = bool(fast_mode)
        return self

    def _load_clip(self):
        """Lazy-load CLIP once. Never raises: failures flip to 'unavailable'.

        Two guards keep this off the request hot path:
          * an in-flight load makes concurrent requests fall back to the
            heuristic immediately instead of queueing behind a download;
          * weights are first looked up in the local HF cache
            (``local_files_only``) so an offline box fails in milliseconds
            rather than hanging on a network timeout.
        """
        if self._clip_state in ("ready", "unavailable"):
            return self._clip_state == "ready"
        if self._clip_state == "loading":
            return False
        with self._lock:
            if self._clip_state in ("ready", "unavailable"):
                return self._clip_state == "ready"
            if self._clip_state == "loading":
                return False
            self._clip_state = "loading"
            try:
                from transformers import CLIPModel, CLIPProcessor

                kwargs = {"cache_dir": self.cache_dir} if self.cache_dir else {}
                try:
                    processor = CLIPProcessor.from_pretrained(
                        self.model_id, local_files_only=True, **kwargs)
                    model = CLIPModel.from_pretrained(
                        self.model_id, local_files_only=True, **kwargs)
                except Exception:
                    if not ATTRIBUTES_CLIP_ALLOW_DOWNLOAD:
                        raise
                    processor = CLIPProcessor.from_pretrained(self.model_id, **kwargs)
                    model = CLIPModel.from_pretrained(self.model_id, **kwargs)

                self._clip_processor = processor
                self._clip_model = model.eval().to(self.device)
                if self.device.type == "cuda":
                    self._clip_model.half()
                self._clip_state = "ready"
                self._clip_error = None
            except Exception as exc:  # offline / missing package / bad weights
                self._clip_model = None
                self._clip_processor = None
                self._clip_error = f"{type(exc).__name__}: {exc}"
                self._clip_state = "unavailable"
            return self._clip_state == "ready"

    def preload(self):
        """Warm the CLIP engine explicitly (e.g. at server start-up)."""
        return self._load_clip()

    @torch.no_grad()
    def _clip_probs(self, image, prompts):
        """Softmax over CLIP image-text similarities (temperature-scaled)."""
        inputs = self._clip_processor(
            text=list(prompts), images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device.type == "cuda":
            inputs = {k: (v.half() if v.dtype == torch.float32 else v)
                      for k, v in inputs.items()}
        logits = self._clip_model(**inputs).logits_per_image.float()
        probs = torch.softmax(logits / self.temperature, dim=-1)[0]
        return probs.cpu().tolist()

    # -- public API ------------------------------------------------------
    def predict(self, image, species="Cattle", breed=None, is_ood=False,
                traits=None, trait_details=None, stats=None, fast_mode=None):
        """Return ``{"gender": {...}, "age_group": {...}}``.

        ``traits`` / ``trait_details`` come from ``decode_traits``; ``stats``
        from ``subject_stats``. Missing inputs simply weaken the heuristic.
        """
        species_key = _species_key(species)
        use_fast = self.fast_mode if fast_mode is None else bool(fast_mode)

        if not ATTRIBUTES_ENABLED:
            return self._skipped("Attributes disabled in config")
        if is_ood:
            return self._skipped("Not evaluated: image flagged as out-of-distribution")

        stats = stats or subject_stats(image)

        if self.clip_enabled and not use_fast and self._load_clip():
            try:
                return self._clip_predict(image, species_key)
            except Exception as exc:  # runtime failure -> heuristic, never 500
                self._clip_error = f"{type(exc).__name__}: {exc}"

        return self._heuristic(species_key, breed, traits or {},
                               trait_details or {}, stats)

    # -- engine 1: CLIP --------------------------------------------------
    def _clip_predict(self, image, species_key):
        gender_prompts = [GENDER_PROMPTS[k].format(species=species_key)
                          for k in ("female", "male")]
        age_prompts = [AGE_PROMPTS[k].format(species=species_key)
                       for k in ("calf", "young", "adult")]

        g_probs = self._clip_probs(image, gender_prompts)
        a_probs = self._clip_probs(image, age_prompts)

        g_key = "female" if g_probs[0] >= g_probs[1] else "male"
        a_key = ("calf", "young", "adult")[int(np.argmax(a_probs))]

        return {
            "gender": {
                "label": GENDER_LABELS[species_key][g_key],
                "short_label": GENDER_SHORT[g_key],
                "key": g_key,
                "confidence": round(max(g_probs) * 100, 1),
                "distribution": {k: round(p * 100, 2) for k, p in
                                 zip(("female", "male"), g_probs)},
                "method": "clip_zeroshot",
                "note": GENDER_NOTE_CLIP,
            },
            "age_group": {
                "label": AGE_LABELS[a_key],
                "short_label": AGE_SHORT[a_key],
                "key": a_key,
                "confidence": round(max(a_probs) * 100, 1),
                "distribution": {k: round(p * 100, 2) for k, p in
                                 zip(("calf", "young", "adult"), a_probs)},
                "method": "clip_zeroshot",
                "note": AGE_NOTE_CLIP,
            },
            "attributes_meta": {
                "engine": "clip_zeroshot",
                "clip": self.clip_status,
                "fast_mode": False,
            },
        }

    # -- engine 2: morphological / breed-prior heuristic -----------------
    def _heuristic(self, species_key, breed, traits, trait_details, stats):
        tokens = {}
        for field in ("hump", "horn", "dewlap", "size"):
            detail = trait_details.get(field) or {}
            token = detail.get("token")
            if not token and traits.get(field):
                token = str(traits[field]).lower().replace(" ", "_")
            tokens[field] = token

        is_dairy = _breed_key(breed) in DAIRY_BREEDS

        # --- gender: log-odds for "female" vs "male" --------------------
        female, male = 1.45, 0.0            # ~80% female dairy-image prior
        if species_key == "buffalo":
            female += 0.15
        if is_dairy:
            female += 0.45

        hump = tokens.get("hump")
        if hump in ("large", "large_well_developed", "well_developed"):
            male += 0.85
        elif hump in ("absent", "small"):
            female += 0.35

        dewlap = tokens.get("dewlap")
        if dewlap == "large_pendulous":
            male += 0.65
        elif dewlap in ("minimal", "small"):
            female += 0.20

        size = tokens.get("size")
        if size in ("large", "medium_large"):
            male += 0.70
        elif size in ("small", "small_medium"):
            female += 0.45

        aspect = stats.get("aspect_ratio")
        if aspect is not None and aspect < 1.2 and stats.get("source") == "foreground":
            female += 0.10                  # compact/upright framing, weak cue

        g_scores = np.array([female, male], dtype=np.float64)
        g_probs = _softmax(g_scores)
        g_key = "female" if g_probs[0] >= g_probs[1] else "male"
        g_conf = self._shrink(max(g_probs), classes=2, ceiling=0.88)

        # --- age: log-odds for calf / young / adult ---------------------
        calf, young, adult = -1.10, 0.15, 1.30   # adult-dominated datasets
        if size == "small":
            calf += 1.55
            young += 0.30
            adult -= 0.60
        elif size == "small_medium":
            calf += 0.55
            young += 1.15
            adult -= 0.35
        elif size == "medium":
            young += 0.55
        elif size in ("medium_large", "large"):
            adult += 0.55
            calf -= 0.35

        if hump in ("small", "absent"):
            calf += 0.25
            young += 0.15
        elif hump in ("large", "large_well_developed"):
            adult += 0.30

        horn = tokens.get("horn")
        if horn in ("absent", "short", "short_stumpy", "stumpy"):
            calf += 0.15
        elif horn in ("long_curved", "long_backward_curved", "heavy_drooping",
                      "semicircular", "lyre_shaped"):
            adult += 0.30

        if dewlap in ("minimal", "small"):
            calf += 0.20
        elif dewlap in ("large_pendulous", "moderate_pendulous"):
            adult += 0.15

        if aspect is not None and aspect < 1.15 and stats.get("source") == "foreground":
            calf += 0.30                    # tall compact silhouette

        a_scores = np.array([calf, young, adult], dtype=np.float64)
        a_probs = _softmax(a_scores)
        a_key = ("calf", "young", "adult")[int(np.argmax(a_probs))]
        a_conf = self._shrink(max(a_probs), classes=3, ceiling=0.82)

        return {
            "gender": {
                "label": GENDER_LABELS[species_key][g_key],
                "short_label": GENDER_SHORT[g_key],
                "key": g_key,
                "confidence": round(g_conf * 100, 1),
                "distribution": {k: round(p * 100, 2) for k, p in
                                 zip(("female", "male"), g_probs)},
                "method": "morphological_heuristic",
                "note": GENDER_NOTE_HEURISTIC,
                "requires_confirmation": True,
            },
            "age_group": {
                "label": AGE_LABELS[a_key],
                "short_label": AGE_SHORT[a_key],
                "key": a_key,
                "confidence": round(a_conf * 100, 1),
                "distribution": {k: round(p * 100, 2) for k, p in
                                 zip(("calf", "young", "adult"), a_probs)},
                "method": "morphological_heuristic",
                "note": AGE_NOTE_HEURISTIC,
                "requires_confirmation": True,
            },
            "attributes_meta": {
                "engine": "morphological_heuristic",
                "clip": self.clip_status,
                "fast_mode": bool(self.fast_mode),
                "inputs": {
                    "species": species_key,
                    "breed": breed,
                    "dairy_prior": is_dairy,
                    "traits_used": {k: v for k, v in tokens.items() if v},
                    "subject": stats,
                },
            },
        }

    @staticmethod
    def _shrink(prob, classes=2, ceiling=0.85):
        """Pull a prior-driven probability toward the uniform baseline.

        The heuristic is a prior, not a measurement; reporting 0.97 would tell
        the UI there is nothing to confirm. Blending toward ``1/classes`` keeps
        the number honest for both the binary (gender) and 3-way (age) heads.
        """
        uniform = 1.0 / max(1, classes)
        return min(ceiling, uniform + (float(prob) - uniform) * 0.6)

    @staticmethod
    def _skipped(reason):
        block = {"label": "Unknown", "short_label": "Unknown", "confidence": 0.0,
                 "method": "skipped", "note": reason}
        return {"gender": dict(block), "age_group": dict(block),
                "attributes_meta": {"engine": "skipped", "reason": reason}}


# ---------------------------------------------------------------------------
#  Process-wide singleton (CLIP weights must not load per request)
# ---------------------------------------------------------------------------

_ESTIMATOR = None
_ESTIMATOR_LOCK = threading.Lock()


def get_estimator(**kwargs):
    """Return (and lazily create) the process-wide AttributeEstimator."""
    global _ESTIMATOR
    if _ESTIMATOR is None:
        with _ESTIMATOR_LOCK:
            if _ESTIMATOR is None:
                _ESTIMATOR = AttributeEstimator(**kwargs)
    return _ESTIMATOR


def estimate_attributes(image, species="Cattle", breed=None, is_ood=False,
                        traits=None, trait_details=None, stats=None,
                        fast_mode=None, **estimator_kwargs):
    """Convenience wrapper around the shared estimator (no CLIP reload)."""
    return get_estimator(**estimator_kwargs).predict(
        image, species=species, breed=breed, is_ood=is_ood, traits=traits,
        trait_details=trait_details, stats=stats, fast_mode=fast_mode)


def build_attribute_report(image, species="Cattle", breed=None, is_ood=False,
                           trait_logits=None, trait_vocab=None, trait_meta=None,
                           fast_mode=None, stats=None, clip_enabled=None,
                           enabled=True):
    """Build the three API blocks (traits, gender, age) in one call.

    Shared by ``test_model.py`` and ``server/services/inference.py`` so the two
    surfaces cannot drift. ``trait_logits`` is ``{field: logits}`` from
    ``src.model.predict_traits`` (empty for ONNX / older checkpoints).
    """
    traits, trait_details = {}, {}
    if trait_logits and not is_ood:
        traits, trait_details = decode_traits(trait_logits, trait_vocab or {})

    if not enabled:
        report = AttributeEstimator._skipped("Attributes disabled in config")
    else:
        estimator = get_estimator().configure(clip_enabled=clip_enabled,
                                              fast_mode=fast_mode)
        report = estimator.predict(
            image, species=species, breed=breed, is_ood=is_ood,
            traits=traits, trait_details=trait_details, stats=stats,
            fast_mode=fast_mode,
        )

    report["traits"] = traits
    report["traits_detail"] = trait_details
    report["traits_meta"] = trait_meta or {
        "available": bool(trait_logits),
        "reason": None if trait_logits else "no_trait_heads",
    }
    return report


def skipped_attribute_report(reason):
    """Public helper for callers that disable attributes entirely."""
    report = AttributeEstimator._skipped(reason)
    report["traits"] = {}
    report["traits_detail"] = {}
    report["traits_meta"] = {"available": False, "reason": reason}
    return report

