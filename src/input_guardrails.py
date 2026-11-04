"""100% offline 4-layer input guardrail / multi-stage OOD engine.

Why this exists
---------------
``OODDetector`` only looks at logits (energy / MSP / binary floor). A confident
CNN can still accept a cartoon, a selfie, a dog or a screenshot. This module adds
three cheap visual/semantic gates plus a feature-space entropy gate **without
retraining the BreedClassifier**:

  Stage 1  photorealism / drawing / screenshot  (NumPy + PIL)
  Stage 2  human / selfie / other-animal gate   (OpenCV Haar + optional CLIP)
  Stage 3  soft-routed entropy, margin, norm    (model tensors)
  Stage 4  the existing logit OODDetector       (energy / MSP / binary floor)

Every rejection carries a localized ``rejection_info`` object (en / ta / hi), so
the API/UI can explain exactly why the image was refused instead of showing a
generic "out of distribution" banner.

Nothing here downloads anything at import time. CLIP is lazy-loaded through the
existing process-wide :class:`~src.attributes_zeroshot.AttributeEstimator`, which
already respects the offline/cache-only configuration.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import (
    GUARDRAIL_CLIP_ENABLED,
    GUARDRAIL_CLIP_VALID_MIN,
    GUARDRAIL_COLOR_ENTROPY_MIN,
    GUARDRAIL_COLOR_TOP3_RATIO,
    GUARDRAIL_ENABLED,
    GUARDRAIL_ENTROPY_MAX,
    GUARDRAIL_FACE_AREA_RATIO,
    GUARDRAIL_FEATURE_NORM_MAX,
    GUARDRAIL_FEATURE_NORM_MIN,
    GUARDRAIL_FLAT_BLOCK,
    GUARDRAIL_FLAT_BLOCK_RATIO,
    GUARDRAIL_FLAT_STD,
    GUARDRAIL_PHOTOREALISM_ENABLED,
    GUARDRAIL_SKIN_CENTER_RATIO,
    GUARDRAIL_SKIN_RATIO,
    GUARDRAIL_SKETCH_EDGE_MIN,
    GUARDRAIL_SKETCH_SAT_MAX,
    GUARDRAIL_SKETCH_WHITE_RATIO,
    GUARDRAIL_TOP1_MIN,
)

LANGS = ("en", "ta", "hi")

STAGE_NAMES = {
    1: "Stage 1: Photorealism Check",
    2: "Stage 2: Human / Semantic Check",
    3: "Stage 3: Feature & Entropy Check",
    4: "Stage 4: Logit OOD Check",
}

# Each reason code carries the exact farmer-facing copy in all three languages.
_REJECTION_COPY = {
    "cartoon_or_digital_graphic": {
        "title": {
            "en": "Drawing / Digital Graphic Detected",
            "ta": "வரைபடம் / டிஜிட்டல் கிராபிக் கண்டறியப்பட்டது",
            "hi": "चित्र / डिजिटल ग्राफ़िक पाया गया",
        },
        "message": {
            "en": "This looks like a cartoon, clip-art, screenshot or digital illustration, not a real photograph.",
            "ta": "இது கார்ட்டூன், கிளிப்-ஆர்ட், ஸ்கிரீன்ஷாட் அல்லது டிஜிட்டல் விளக்கம் போல் தெரிகிறது; உண்மையான புகைப்படம் அல்ல.",
            "hi": "यह कार्टून, क्लिप-आर्ट, स्क्रीनशॉट या डिजिटल चित्र जैसा है, वास्तविक फोटो नहीं।",
        },
        "instruction": {
            "en": "Take a new photo of the live animal with the rear camera in daylight.",
            "ta": "பகல் வெளிச்சத்தில் பின்பக்க கேமராவால் விலங்கின் புதிய புகைப்படத்தை எடுக்கவும்.",
            "hi": "दिन के उजाले में पिछले कैमरे से पशु की नई फोटो लें।",
        },
    },
    "sketch_or_line_drawing": {
        "title": {
            "en": "Sketch / Line Drawing Detected",
            "ta": "ஓவியம் / கோட்டு வரைபடம் கண்டறியப்பட்டது",
            "hi": "स्केच / रेखा चित्र पाया गया",
        },
        "message": {
            "en": "Please upload a real photograph. Sketches, line art and drawings cannot be classified.",
            "ta": "தயவுசெய்து உண்மையான புகைப்படத்தை பதிவேற்றவும். ஓவியங்கள் மற்றும் கோட்டு வரைபடங்களை வகைப்படுத்த முடியாது.",
            "hi": "कृपया वास्तविक फोटो अपलोड करें। स्केच, रेखा चित्र और चित्र वर्गीकृत नहीं किए जा सकते।",
        },
        "instruction": {
            "en": "Photograph the live animal from the side, filling most of the frame.",
            "ta": "விலங்கின் பக்கவாட்டில் இருந்து, பெரும்பாலான பகுதியை நிரப்பி புகைப்படம் எடுக்கவும்.",
            "hi": "पशु की बगल से, फ्रेम का अधिकांश भाग भरते हुए फोटो लें।",
        },
    },
    "synthetic_illustration": {
        "title": {
            "en": "Illustration / Synthetic Image Detected",
            "ta": "விளக்கப்படம் / செயற்கை படம் கண்டறியப்பட்டது",
            "hi": "चित्रण / कृत्रिम छवि पाया गया",
        },
        "message": {
            "en": "The image has too few natural colours to be a real camera photo.",
            "ta": "இயற்கையான நிறங்கள் மிகக் குறைவாக இருப்பதால் இது உண்மையான கேமரா புகைப்படமாக இருக்க வாய்ப்பில்லை.",
            "hi": "प्राकृतिक रंग बहुत कम हैं, इसलिए यह वास्तविक कैमरा फोटो नहीं लगती।",
        },
        "instruction": {
            "en": "Use the rear camera and avoid screenshots, posters or edited images.",
            "ta": "பின்பக்க கேமராவைப் பயன்படுத்தவும்; ஸ்கிரீன்ஷாட், போஸ்டர் அல்லது திருத்தப்பட்ட படங்களைத் தவிர்க்கவும்.",
            "hi": "पिछला कैमरा इस्तेमाल करें; स्क्रीनशॉट, पोस्टर या संपादित तस्वीरें न लें।",
        },
    },
    "human_or_selfie": {
        "title": {
            "en": "Human / Selfie Image Detected",
            "ta": "மனித புகைப்படம் / செல்ஃபி கண்டறியப்பட்டது",
            "hi": "मानव फोटो / सेल्फी का पता चला",
        },
        "message": {
            "en": "Please upload a real photograph of a Cattle or Buffalo. Human portraits and selfies are not allowed.",
            "ta": "தயவுசெய்து மாடு அல்லது எருமையின் உண்மையான புகைப்படத்தை பதிவேற்றவும். மனித புகைப்படங்கள் அனுமதிக்கப்படாது.",
            "hi": "कृपया गाय या भैंस की वास्तविक फोटो अपलोड करें। इंसानों की फोटो या सेल्फी मान्य नहीं है।",
        },
        "instruction": {
            "en": "Stand 2–3 meters away and capture the full side profile of the animal in daylight.",
            "ta": "பகல் வெளிச்சத்தில் மாட்டின் முழு பக்கவாட்டு தோற்றத்தை 2-3 மீட்டர் தொலைவில் இருந்து எடுக்கவும்.",
            "hi": "दिन के उजाले में 2-3 मीटर की दूरी से पशु के पूरे साइड प्रोफाइल की फोटो लें।",
        },
    },
    "other_animal": {
        "title": {
            "en": "Another Animal Detected",
            "ta": "வேறு விலங்கு கண்டறியப்பட்டது",
            "hi": "कोई अन्य पशु पाया गया",
        },
        "message": {
            "en": "The image appears to show a dog, goat, sheep or another animal, not cattle or buffalo.",
            "ta": "இந்தப் படத்தில் நாய், ஆடு, செம்மறி அல்லது வேறு விலங்கு இருப்பதாகத் தெரிகிறது; மாடு அல்லது எருமை அல்ல.",
            "hi": "इस छवि में कुत्ता, बकरी, भेड़ या कोई अन्य पशु लगता है, गाय या भैंस नहीं।",
        },
        "instruction": {
            "en": "Point the camera only at a cow, bull, ox, calf or buffalo.",
            "ta": "கேமராவை மாடு, காளை, எரு, கன்று அல்லது எருமை மீது மட்டும் காட்டவும்.",
            "hi": "कैमरा केवल गाय, बैल, बछड़ा या भैंस की ओर रखें।",
        },
    },
    "unrelated_object": {
        "title": {
            "en": "No Cattle or Buffalo Detected",
            "ta": "மாடு அல்லது எருமை கண்டறியப்படவில்லை",
            "hi": "कोई गाय या भैंस नहीं मिली",
        },
        "message": {
            "en": "The image does not appear to contain a cow, bull, ox, calf or buffalo.",
            "ta": "இந்தப் படத்தில் மாடு, காளை, எரு, கன்று அல்லது எருமை இருப்பதாகத் தெரியவில்லை.",
            "hi": "इस छवि में गाय, बैल, बछड़ा या भैंस नहीं दिख रही।",
        },
        "instruction": {
            "en": "Fill the frame with the animal and avoid vehicles, buildings, documents or empty scenery.",
            "ta": "விலங்கை மட்டும் படத்தில் நிரப்பவும்; வாகனங்கள், கட்டிடங்கள், ஆவணங்கள் அல்லது வெற்று இயற்கைக் காட்சிகளைத் தவிர்க்கவும்.",
            "hi": "फ्रेम में केवल पशु रखें; वाहन, इमारत, दस्तावेज़ या खाली दृश्य न लें।",
        },
    },
    "uncertain_animal": {
        "title": {
            "en": "Animal Not Recognized",
            "ta": "விலங்கு அடையாளம் காணப்படவில்லை",
            "hi": "पशु पहचाना नहीं गया",
        },
        "message": {
            "en": "An animal may be present, but its breed features are unclear or the image is too ambiguous.",
            "ta": "விலங்கு இருக்கலாம், ஆனால் அதன் இன அம்சங்கள் தெளிவாக இல்லை அல்லது படம் மிகவும் தெளிவற்றது.",
            "hi": "पशु हो सकता है, लेकिन उसकी नस्ल की विशेषताएँ स्पष्ट नहीं हैं या छवि बहुत अस्पष्ट है।",
        },
        "instruction": {
            "en": "Move closer, fill the frame with the full side profile, and avoid blur or backlight.",
            "ta": "அருகில் சென்று, முழு பக்கவாட்டுத் தோற்றத்தை படத்தில் நிரப்பி, மங்கல் அல்லது பின்னொளியைத் தவிர்க்கவும்.",
            "hi": "पास जाएँ, पूरा साइड प्रोफाइल फ्रेम में भरें और धुंधलापन या बैकलाइट से बचें।",
        },
    },
    "out_of_distribution": {
        "title": {
            "en": "Image Outside the Model’s Knowledge",
            "ta": "மாதிரியின் அறிவுக்கு வெளியே உள்ள படம்",
            "hi": "मॉडल के ज्ञान से बाहर की छवि",
        },
        "message": {
            "en": "The classifier is not confident that this is a cattle or buffalo photograph.",
            "ta": "இது மாடு அல்லது எருமை புகைப்படம் என்று வகைப்படுத்தி உறுதியாக நம்பவில்லை.",
            "hi": "वर्गीकरणकर्ता को भरोसा नहीं है कि यह गाय या भैंस की फोटो है।",
        },
        "instruction": {
            "en": "Upload a clear, well-lit side profile of one animal.",
            "ta": "ஒரு விலங்கின் தெளிவான, நல்ல வெளிச்சத்தில் எடுக்கப்பட்ட பக்கவாட்டுப் படத்தை பதிவேற்றவும்.",
            "hi": "एक पशु की स्पष्ट, अच्छी रोशनी वाली साइड प्रोफाइल फोटो अपलोड करें।",
        },
    },
}

# CLIP semantic anchors (one prompt per category).
_CLIP_CATEGORIES = [
    ("valid", None, "a real camera photo of a live cow, bull, ox, calf, or water buffalo"),
    ("human", "human_or_selfie", "a selfie, portrait, or photo of a human person, man, woman, or child"),
    ("art", "sketch_or_line_drawing", "a drawing, cartoon, sketch, painting, vector illustration, or 3d render of an animal"),
    ("other_animal", "other_animal", "a photo of a dog, cat, goat, sheep, horse, pig, deer, elephant, or bird"),
    ("object", "unrelated_object", "a photo of a car, building, room, food, document, screenshot, or empty landscape"),
]


def _to_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    try:
        return torch.as_tensor(np.asarray(value), dtype=torch.float32)
    except Exception:
        return None


def _first_batch(value: Any) -> Optional[torch.Tensor]:
    tensor = _to_tensor(value)
    if tensor is None:
        return None
    if tensor.dim() >= 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor


class InputGuardrailEngine:
    """Fast, offline 4-layer guardrail shared by the API and the GUI."""

    def __init__(self, enabled: Optional[bool] = None,
                 clip_enabled: Optional[bool] = None,
                 device: Any = None):
        self.enabled = GUARDRAIL_ENABLED if enabled is None else bool(enabled)
        self.clip_enabled = (GUARDRAIL_CLIP_ENABLED if clip_enabled is None
                             else bool(clip_enabled))
        self.device = device
        self._cv2 = None
        self._face_cascade = None
        self._profile_cascade = None
        self._cascade_state = "unloaded"

    # -- status -----------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        clip_state = "disabled"
        if self.clip_enabled:
            try:
                from .attributes_zeroshot import get_estimator
                clip_state = get_estimator().clip_status.get("state", "unknown")
            except Exception:
                clip_state = "unavailable"
        return {
            "enabled": self.enabled,
            "opencv": self._load_cascades(),
            "clip": clip_state,
            "thresholds": {
                "flat_block_ratio": GUARDRAIL_FLAT_BLOCK_RATIO,
                "flat_std": GUARDRAIL_FLAT_STD,
                "color_entropy_min": GUARDRAIL_COLOR_ENTROPY_MIN,
                "face_area_ratio": GUARDRAIL_FACE_AREA_RATIO,
                "entropy_max": GUARDRAIL_ENTROPY_MAX,
                "top1_min": GUARDRAIL_TOP1_MIN,
                "feature_norm": [GUARDRAIL_FEATURE_NORM_MIN, GUARDRAIL_FEATURE_NORM_MAX],
            },
        }

    # -- public API -------------------------------------------------------
    def validate_image(self, image_bytes: Optional[bytes], pil_img: Optional[Image.Image],
                       model_outputs: Optional[Dict[str, Any]] = None,
                       ood_detector: Any = None) -> Dict[str, Any]:
        """Run all available stages and return a structured verdict."""
        started = time.perf_counter()
        if not self.enabled:
            return self._accepted({}, started, skipped=True)

        image = pil_img
        if image is None and image_bytes:
            try:
                import io as _io
                image = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                image = None

        stages: Dict[str, Any] = {}
        if image is not None:
            visual = self.validate_visual(image)
            stages["stage1_2"] = visual
            if visual["rejected"]:
                return self._rejected(visual, started, stages)
        else:
            stages["stage1_2"] = {"rejected": False, "details": {"skipped": "no image"}}

        if model_outputs is not None:
            model_verdict = self.validate_model_outputs(model_outputs, ood_detector)
            stages["stage3_4"] = model_verdict
            if model_verdict["rejected"]:
                return self._rejected(model_verdict, started, stages)
            details = model_verdict.get("details", {})
        else:
            details = {}

        return self._accepted(details, started, stages=stages)

    def validate_visual(self, pil_img: Image.Image) -> Dict[str, Any]:
        """Stages 1–2 only (safe to run before the classifier)."""
        if not self.enabled:
            return {"rejected": False, "details": {"skipped": "disabled"}}

        started = time.perf_counter()
        if GUARDRAIL_PHOTOREALISM_ENABLED:
            realism = self._check_photorealism(pil_img)
            if realism["rejected"]:
                realism["stage"] = STAGE_NAMES[1]
                realism["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                return self._rejected(realism, started, {})

        human = self._check_human_or_selfie(pil_img)
        if human["rejected"]:
            human["stage"] = STAGE_NAMES[2]
            human["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return self._rejected(human, started, {})
        return self._accepted(human.get("details", {}), started)

    def validate_model_outputs(self, model_outputs: Dict[str, Any],
                               ood_detector: Any = None) -> Dict[str, Any]:
        """Stages 3–4 using the classifier's tensors."""
        if not self.enabled:
            return {"rejected": False, "details": {"skipped": "disabled"}}
        started = time.perf_counter()
        feature = self._check_feature_and_entropy(model_outputs)
        if feature["rejected"]:
            feature["stage"] = STAGE_NAMES[3]
            feature["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return self._rejected(feature, started, {})

        if ood_detector is not None:
            logit = self._check_logit_ood(model_outputs, ood_detector)
            if logit["rejected"]:
                logit["stage"] = STAGE_NAMES[4]
                logit["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                return self._rejected(logit, started, {})

        return self._accepted(feature.get("details", {}), started)

    # -- Stage 1 ----------------------------------------------------------
    def _check_photorealism(self, pil_img: Image.Image) -> Dict[str, Any]:
        image = pil_img.convert("RGB").resize((256, 256), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32)

        # 1a. Flat-colour blocks (cartoon / clip-art / screenshot).
        block = max(2, int(GUARDRAIL_FLAT_BLOCK))
        h, w = arr.shape[:2]
        gh, gw = h // block, w // block
        gray = arr.mean(axis=2)
        blocks = gray[:gh * block, :gw * block].reshape(gh, block, gw, block)
        blocks = blocks.swapaxes(1, 2).reshape(gh * gw, block * block)
        stds = blocks.std(axis=1)
        means = blocks.mean(axis=1)
        # Ignore near-white/near-black flat blocks so paper/line-art canvases
        # fall through to the sketch check below instead of being mislabelled
        # as clip-art.
        flat = (stds < GUARDRAIL_FLAT_STD) & (means < 245) & (means > 12)
        flat_ratio = float(flat.mean())
        metrics = {"flat_block_ratio": round(flat_ratio, 4)}

        if flat_ratio > GUARDRAIL_FLAT_BLOCK_RATIO:
            return {
                "rejected": True,
                "reason_code": "cartoon_or_digital_graphic",
                "details": metrics,
            }

        # 1b. Sketch / pure white or black canvas with sharp edges.
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
        sat_mean = float(hsv[:, :, 1].mean() / 255.0)
        white = float(((arr[:, :, 0] > 245) & (arr[:, :, 1] > 245)
                       & (arr[:, :, 2] > 245)).mean())
        black = float(((arr[:, :, 0] < 12) & (arr[:, :, 1] < 12)
                       & (arr[:, :, 2] < 12)).mean())
        gy, gx = np.gradient(gray / 255.0)
        edge_ratio = float((np.hypot(gx, gy) > 0.25).mean())
        metrics.update({
            "saturation_mean": round(sat_mean, 4),
            "white_ratio": round(white, 4),
            "black_ratio": round(black, 4),
            "edge_ratio": round(edge_ratio, 4),
        })
        if ((white > GUARDRAIL_SKETCH_WHITE_RATIO
             or black > GUARDRAIL_SKETCH_WHITE_RATIO)
                and sat_mean < GUARDRAIL_SKETCH_SAT_MAX
                and edge_ratio > GUARDRAIL_SKETCH_EDGE_MIN):
            return {
                "rejected": True,
                "reason_code": "sketch_or_line_drawing",
                "details": metrics,
            }

        # 1c. Quantized 4x4x4 colour-histogram entropy.
        quant = np.clip((arr // 64).astype(np.int32), 0, 3)
        index = quant[:, :, 0] * 16 + quant[:, :, 1] * 4 + quant[:, :, 2]
        hist = np.bincount(index.ravel(), minlength=64).astype(np.float64)
        hist /= max(1.0, hist.sum())
        nz = hist[hist > 0]
        entropy = float(-(nz * np.log2(nz)).sum())
        top3 = float(np.sort(hist)[-3:].sum())
        metrics.update({
            "color_entropy": round(entropy, 4),
            "top3_color_ratio": round(top3, 4),
        })
        if (entropy < GUARDRAIL_COLOR_ENTROPY_MIN
                or top3 > GUARDRAIL_COLOR_TOP3_RATIO):
            return {
                "rejected": True,
                "reason_code": "synthetic_illustration",
                "details": metrics,
            }

        return {"rejected": False, "details": metrics}

    # -- Stage 2 ----------------------------------------------------------
    def _load_cascades(self) -> bool:
        if self._cascade_state in ("ready", "unavailable"):
            return self._cascade_state == "ready"
        try:
            import cv2
            base = cv2.data.haarcascades
            self._face_cascade = cv2.CascadeClassifier(
                base + "haarcascade_frontalface_default.xml")
            self._profile_cascade = cv2.CascadeClassifier(
                base + "haarcascade_profileface.xml")
            self._cv2 = cv2
            self._cascade_state = "ready"
        except Exception:
            self._cv2 = None
            self._face_cascade = None
            self._profile_cascade = None
            self._cascade_state = "unavailable"
        return self._cascade_state == "ready"

    def _check_human_or_selfie(self, pil_img: Image.Image) -> Dict[str, Any]:
        image = pil_img.convert("RGB")
        arr = np.asarray(image, dtype=np.uint8)
        total = float(arr.shape[0] * arr.shape[1])
        details: Dict[str, Any] = {}

        # 2a. Haar face / profile face (if OpenCV is available).
        face_ratio = 0.0
        if self._load_cascades():
            gray = self._cv2.cvtColor(arr, self._cv2.COLOR_RGB2GRAY)
            faces = []
            for cascade in (self._face_cascade, self._profile_cascade):
                if cascade is None or cascade.empty():
                    continue
                found = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(found):
                    faces.extend(found)
            if faces:
                face_ratio = max(float(w * h) / total for (_, _, w, h) in faces)
        details["face_area_ratio"] = round(face_ratio, 4)
        if face_ratio >= GUARDRAIL_FACE_AREA_RATIO:
            return {
                "rejected": True,
                "reason_code": "human_or_selfie",
                "details": details,
            }

        # 2b. Skin-tone portrait heuristic (conservative fallback).
        try:
            ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.uint8)
            y, cb, cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
            skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
                    & (y > 60))
            skin_ratio = float(skin.mean())
            h, w = skin.shape
            center = skin[h // 4: (3 * h) // 4, w // 4: (3 * w) // 4]
            center_ratio = float(center.mean()) if center.size else 0.0
            details["skin_ratio"] = round(skin_ratio, 4)
            details["center_skin_ratio"] = round(center_ratio, 4)
            if (skin_ratio > GUARDRAIL_SKIN_RATIO
                    and center_ratio > GUARDRAIL_SKIN_CENTER_RATIO):
                return {
                    "rejected": True,
                    "reason_code": "human_or_selfie",
                    "details": details,
                }
        except Exception:
            pass

        # 2c. CLIP semantic gate (uses the already configured offline CLIP).
        clip = self._check_clip_semantics(image)
        if clip is not None:
            details["clip_probs"] = clip["probs"]
            details["clip_winner"] = clip["winner"]
            if clip["reason_code"] is not None:
                return {
                    "rejected": True,
                    "reason_code": clip["reason_code"],
                    "details": details,
                }

        return {"rejected": False, "details": details}

    def _check_clip_semantics(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        if not self.clip_enabled:
            return None
        try:
            from .attributes_zeroshot import get_estimator
            estimator = get_estimator()
            if not estimator._load_clip():
                return None
            probs = estimator._clip_probs(image, [item[2] for item in _CLIP_CATEGORIES])
        except Exception:
            return None

        winner_idx = int(np.argmax(probs))
        winner = _CLIP_CATEGORIES[winner_idx][0]
        valid_conf = float(probs[0])
        if winner_idx == 0 and valid_conf >= GUARDRAIL_CLIP_VALID_MIN:
            return {"winner": winner, "probs": probs, "reason_code": None}
        reason = _CLIP_CATEGORIES[winner_idx][1]
        if reason is None:
            reason = "uncertain_animal"
        return {"winner": winner, "probs": probs, "reason_code": reason}

    # -- Stage 3 ----------------------------------------------------------
    def _check_feature_and_entropy(self, model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        binary = _first_batch(model_outputs.get("binary"))
        cattle = _first_batch(model_outputs.get("cattle"))
        buffalo = _first_batch(model_outputs.get("buffalo"))
        if binary is None or cattle is None or buffalo is None:
            return {"rejected": False, "details": {"skipped": "missing logits"}}

        p_binary = F.softmax(binary, dim=-1)
        p_cattle = F.softmax(cattle, dim=-1)
        p_buffalo = F.softmax(buffalo, dim=-1)
        combined = torch.cat([p_binary[0] * p_cattle, p_binary[1] * p_buffalo])
        total = int(combined.numel())
        entropy = float(-(combined * torch.log(combined + 1e-12)).sum()
                        / np.log(max(2, total)))
        top = torch.topk(combined, k=min(2, total)).values
        top1 = float(top[0])
        top2 = float(top[1]) if top.numel() > 1 else 0.0
        margin = (top1 - top2) / (top1 + 1e-12)

        feature = _first_batch(model_outputs.get("features"))
        norm = float(torch.linalg.vector_norm(feature)) if feature is not None else None

        details = {
            "entropy": round(entropy, 4),
            "top1": round(top1, 4),
            "top2": round(top2, 4),
            "margin_ratio": round(float(margin), 4),
            "feature_norm": round(norm, 4) if norm is not None else None,
            "classes": total,
        }

        if entropy > GUARDRAIL_ENTROPY_MAX:
            return {"rejected": True, "reason_code": "uncertain_animal",
                    "details": {**details, "trigger": "entropy"}}
        if top1 < GUARDRAIL_TOP1_MIN:
            return {"rejected": True, "reason_code": "uncertain_animal",
                    "details": {**details, "trigger": "top1"}}
        if norm is not None and not (GUARDRAIL_FEATURE_NORM_MIN <= norm <= GUARDRAIL_FEATURE_NORM_MAX):
            return {"rejected": True, "reason_code": "uncertain_animal",
                    "details": {**details, "trigger": "feature_norm"}}
        return {"rejected": False, "details": details}

    # -- Stage 4 ----------------------------------------------------------
    def _check_logit_ood(self, model_outputs: Dict[str, Any],
                         ood_detector: Any) -> Dict[str, Any]:
        try:
            binary = _to_tensor(model_outputs.get("binary"))
            cattle = _to_tensor(model_outputs.get("cattle"))
            buffalo = _to_tensor(model_outputs.get("buffalo"))
            if binary is None or cattle is None or buffalo is None:
                return {"rejected": False, "details": {"skipped": "missing logits"}}
            if binary.dim() == 1:
                binary = binary.unsqueeze(0)
            if cattle.dim() == 1:
                cattle = cattle.unsqueeze(0)
            if buffalo.dim() == 1:
                buffalo = buffalo.unsqueeze(0)
            is_ood, score, details = ood_detector.is_ood(binary, cattle, buffalo)
        except Exception as exc:
            return {"rejected": False, "details": {"error": str(exc)}}
        if is_ood:
            return {"rejected": True, "reason_code": "out_of_distribution",
                    "details": {**details, "ood_score": score}}
        return {"rejected": False, "details": {**details, "ood_score": score}}

    # -- verdict helpers --------------------------------------------------
    def _rejected(self, verdict: Dict[str, Any], started: float,
                  stages: Dict[str, Any]) -> Dict[str, Any]:
        reason = verdict.get("reason_code") or "out_of_distribution"
        stage = verdict.get("stage") or STAGE_NAMES[3]
        copy = _REJECTION_COPY.get(reason, _REJECTION_COPY["out_of_distribution"])
        info = {
            "rejected": True,
            "reason_code": reason,
            "stage": stage,
            "title": dict(copy["title"]),
            "message": dict(copy["message"]),
            "instruction": dict(copy["instruction"]),
        }
        return {
            "rejected": True,
            "guardrail_triggered": True,
            "reason_code": reason,
            "stage": stage,
            "details": verdict.get("details", {}),
            "rejection_info": info,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "stages": stages,
            "enabled": True,
        }

    def _accepted(self, details: Dict[str, Any], started: float,
                  stages: Optional[Dict[str, Any]] = None,
                  skipped: bool = False) -> Dict[str, Any]:
        return {
            "rejected": False,
            "guardrail_triggered": False,
            "reason_code": None,
            "stage": None,
            "details": details,
            "rejection_info": None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "stages": stages or {},
            "enabled": not skipped,
        }


def verdict_to_ood_explanation(verdict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert a rejected verdict into the legacy ``ood_explanation`` shape."""
    if not verdict or not verdict.get("rejected"):
        return None
    info = verdict.get("rejection_info") or {}
    return {
        "code": info.get("reason_code") or "out_of_distribution",
        "severity": "error",
        "title": dict(info.get("title", {})),
        "message": dict(info.get("message", {})),
        "tips_by_language": {
            lang: [info.get("instruction", {}).get(lang, "")]
            for lang in LANGS
        },
    }


_ENGINE: Optional[InputGuardrailEngine] = None


def get_guardrail_engine(**kwargs) -> InputGuardrailEngine:
    """Return the process-wide guardrail engine."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = InputGuardrailEngine(**kwargs)
    return _ENGINE


__all__ = [
    "InputGuardrailEngine",
    "get_guardrail_engine",
    "verdict_to_ood_explanation",
    "STAGE_NAMES",
]
