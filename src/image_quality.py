"""Pre-inference photo quality gate + farmer-friendly OOD explanations.

Field photos fail in predictable ways: motion blur, dark barns, blown-out
sunlight, tiny uploads, or a crop so wide that the animal is a speck. None of
that needs a model to detect — a 256 px grayscale thumbnail is enough, and the
whole check runs in single-digit milliseconds on CPU.

This module is deliberately dependency-light: Pillow, NumPy and the stdlib
only. It never raises for bad input; an unreadable file returns a report with
``issues=["unreadable"]``.

PUBLIC API
  analyze_image_quality(image_bytes_or_pil) -> report dict
  explain_ood_for_user(is_ood, ood_details, quality_report) -> explanation dict

Every user-facing string is provided in English, Tamil and Hindi. Technical
numbers stay in the payload for developer views but never in the headline.
"""

import io
import time

import numpy as np
from PIL import Image

from .config import (QUALITY_ANALYSIS_SIZE, QUALITY_BLUR_THRESHOLD,
                     QUALITY_DARK_MEAN, QUALITY_HIGHLIGHT_CLIP,
                     QUALITY_MAX_ASPECT, QUALITY_MIN_DIM, QUALITY_SHADOW_CLIP,
                     QUALITY_BRIGHT_MEAN, QUALITY_ENABLED)

LANGS = ("en", "ta", "hi")


# ---------------------------------------------------------------------------
#  Trilingual copy
# ---------------------------------------------------------------------------

TIPS = {
    "blurry": {
        "en": ("Photo is blurry. Hold the phone steady, tap to focus, and take the "
               "side-profile photo again."),
        "ta": ("படம் மங்கலாக உள்ளது. மொபைலை அசைக்காமல் பிடித்து, திரையைத் தட்டி "
               "ஃபோகஸ் செய்து, பக்கவாட்டுப் படத்தை மீண்டும் எடுக்கவும்."),
        "hi": ("फ़ोटो धुंधली है। फ़ोन को स्थिर रखें, फ़ोकस के लिए स्क्रीन पर टैप करें "
               "और साइड-व्यू फ़ोटो दोबारा लें।"),
    },
    "too_dark": {
        "en": ("Photo is slightly dark. For best breed accuracy, take a side-profile "
               "photo outdoors in natural daylight."),
        "ta": ("படம் சற்று இருட்டாக உள்ளது. சிறந்த இனத் துல்லியத்திற்கு, வெளிப்புறத்தில் "
               "இயற்கை வெளிச்சத்தில் பக்கவாட்டுப் படம் எடுக்கவும்."),
        "hi": ("फ़ोटो थोड़ी अंधेरी है। बेहतर नस्ल पहचान के लिए बाहर प्राकृतिक रोशनी में "
               "साइड-व्यू फ़ोटो लें।"),
    },
    "overexposed": {
        "en": ("Photo is too bright / washed out. Avoid direct sun behind the animal "
               "and move to even shade."),
        "ta": ("படம் மிகவும் வெளிச்சமாக உள்ளது. விலங்கின் பின்னால் நேரடி சூரிய ஒளியைத் "
               "தவிர்த்து, சீரான நிழலுக்கு மாறவும்."),
        "hi": ("फ़ोटो बहुत चमकीली है। जानवर के पीछे सीधी धूप से बचें और समान छाया में जाएँ।"),
    },
    "low_resolution": {
        "en": ("Photo resolution is low. Move closer or raise the camera resolution so "
               "the animal fills the frame."),
        "ta": ("படத்தின் தெளிவு குறைவு. விலங்கு சட்டகத்தை நிரப்பும் வகையில் அருகில் செல்லவும் "
               "அல்லது கேமரா தெளிவை அதிகரிக்கவும்."),
        "hi": ("फ़ोटो का रेज़ोल्यूशन कम है। जानवर फ़्रेम में पूरा आए इसके लिए पास जाएँ या "
               "कैमरा रेज़ोल्यूशन बढ़ाएँ।"),
    },
    "extreme_aspect": {
        "en": ("Photo is very wide or very tall. Frame the whole animal side-on so "
               "nothing is cut off."),
        "ta": ("படம் மிக அகலமாக அல்லது மிக உயரமாக உள்ளது. எதுவும் வெட்டப்படாத வகையில் "
               "விலங்கை முழுமையாக பக்கவாட்டில் சட்டகப்படுத்தவும்."),
        "hi": ("फ़ोटो बहुत चौड़ी या बहुत लंबी है। कुछ भी कटे बिना जानवर को पूरा साइड से "
               "फ़्रेम करें।"),
    },
    "unreadable": {
        "en": "This file could not be read as an image. Please upload a JPG or PNG photo.",
        "ta": ("இந்தக் கோப்பை படமாகப் படிக்க முடியவில்லை. JPG அல்லது PNG படத்தைப் "
               "பதிவேற்றவும்."),
        "hi": ("यह फ़ाइल छवि के रूप में नहीं पढ़ी जा सकी। कृपया JPG या PNG फ़ोटो अपलोड करें।"),
    },
    "good_angle": {
        "en": ("Good. Keep the full side profile visible from head to tail, with the "
               "horns, hump and coat unobstructed."),
        "ta": ("நல்லது. தலை முதல் வால் வரை முழு பக்கவாட்டுத் தோற்றம், கொம்பு, திமில் "
               "மற்றும் உடல் நிறம் தெளிவாகத் தெரியுமாறு பிடிக்கவும்."),
        "hi": ("अच्छा। सिर से पूँछ तक पूरा साइड-व्यू दिखाएँ, सींग, कूबड़ और रंग साफ़ रहें।"),
    },
}

MESSAGES = {
    "ok": {
        "title": {"en": "Photo quality looks good",
                  "ta": "படத்தின் தரம் நல்லது",
                  "hi": "फ़ोटो की गुणवत्ता अच्छी है"},
        "message": {"en": "The photo is sharp enough and well exposed for breed analysis.",
                    "ta": ("இனப் பகுப்பாய்வுக்கு படம் போதுமான தெளிவுடனும் சரியான "
                           "வெளிச்சத்திலும் உள்ளது."),
                    "hi": ("नस्ल विश्लेषण के लिए फ़ोटो पर्याप्त स्पष्ट और सही एक्सपोज़र "
                           "में है।")},
        "tips": (),
    },
    "not_animal": {
        "title": {"en": "No cattle or buffalo detected",
                  "ta": "மாடு அல்லது எருமை கண்டறியப்படவில்லை",
                  "hi": "गाय या भैंस नहीं पहचानी गई"},
        "message": {
            "en": ("No cow or buffalo clearly detected in this photo. Please upload a "
                   "clear side-view photo of cattle or buffalo."),
            "ta": ("இந்தப் படத்தில் மாடு அல்லது எருமை தெளிவாகத் தெரியவில்லை. மாடு அல்லது "
                   "எருமையின் தெளிவான பக்கவாட்டுப் படத்தைப் பதிவேற்றவும்."),
            "hi": ("इस फ़ोटो में गाय या भैंस स्पष्ट रूप से नहीं दिखी। कृपया गाय या भैंस की "
                   "स्पष्ट साइड-व्यू फ़ोटो अपलोड करें।"),
        },
        "tips": ("good_angle",),
    },
    "unclear_features": {
        "title": {"en": "Animal detected — breed features unclear",
                  "ta": "விலங்கு கண்டறியப்பட்டது — இன அம்சங்கள் தெளிவில்லை",
                  "hi": "जानवर पहचाना गया — नस्ल की विशेषताएँ अस्पष्ट"},
        "message": {
            "en": ("Animal detected, but breed features (horns, hump, coat) are "
                   "partially hidden or unclear. Try capturing the full side profile "
                   "from head to tail."),
            "ta": ("விலங்கு கண்டறியப்பட்டது, ஆனால் இன அம்சங்கள் (கொம்பு, திமில், உடல் "
                   "நிறம்) முழுமையாகத் தெரியவில்லை. தலை முதல் வால் வரை முழு பக்கவாட்டுப் "
                   "படத்தை எடுக்கவும்."),
            "hi": ("जानवर पहचाना गया, लेकिन नस्ल की विशेषताएँ (सींग, कूबड़, रंग) आंशिक रूप "
                   "से छिपी या अस्पष्ट हैं। सिर से पूँछ तक पूरा साइड-व्यू लेने का प्रयास करें।"),
        },
        "tips": ("good_angle",),
    },
    "poor_quality": {
        "title": {"en": "Photo accepted — improve it for a better result",
                  "ta": "படம் ஏற்கப்பட்டது — சிறந்த முடிவுக்கு மேம்படுத்தவும்",
                  "hi": "फ़ोटो स्वीकार की गई — बेहतर परिणाम के लिए सुधारें"},
        "message": {
            "en": ("The prediction below is the best available, but the photo quality "
                   "limits accuracy. Use the tips to retake it if the breed looks wrong."),
            "ta": ("கீழே உள்ள முடிவு கிடைத்தவற்றில் சிறந்தது, ஆனால் படத்தின் தரம் "
                   "துல்லியத்தைக் குறைக்கிறது. இனம் தவறாகத் தோன்றினால் குறிப்புகளைப் "
                   "பயன்படுத்தி மீண்டும் எடுக்கவும்."),
            "hi": ("नीचे दिया गया परिणाम उपलब्ध सर्वश्रेष्ठ है, लेकिन फ़ोटो की गुणवत्ता "
                   "सटीकता सीमित करती है। नस्ल गलत लगे तो सुझावों के साथ दोबारा फ़ोटो लें।"),
        },
        "tips": (),
    },
}


def _tip(key):
    return dict(TIPS[key])


# Cheap magic-number guard: rejecting junk before PIL avoids a costly exception
# path (~30 ms of traceback handling) inside the /api/predict request.
_IMAGE_SIGNATURES = (
    b"\xff\xd8\xff",      # JPEG
    b"\x89PNG\r\n",       # PNG
    b"GIF87a", b"GIF89a",  # GIF
    b"BM",                # BMP
    b"II*\x00", b"MM\x00*",  # TIFF
)


def _looks_like_image(data):
    head = data[:12]
    if head.startswith(_IMAGE_SIGNATURES):
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    return False


# ---------------------------------------------------------------------------
#  Core metrics
# ---------------------------------------------------------------------------

def _to_gray(source):
    """Return (gray_thumbnail, original_width, original_height).

    Accepts raw bytes or an already-decoded PIL image (the API path passes the
    decoded image to avoid paying for JPEG decode twice).
    """
    if isinstance(source, Image.Image):
        img = source
        width, height = img.size
        orientation = 1
    else:
        data = bytes(source)
        if not data:
            raise ValueError("empty image data")
        if not _looks_like_image(data):
            raise ValueError("unsupported or corrupt image header")
        img = Image.open(io.BytesIO(data))
        width, height = img.size
        # EXIF orientation 5-8 rotate by 90°, so swap for the aspect check.
        try:
            orientation = img.getexif().get(0x0112, 1)
        except Exception:
            orientation = 1
        try:
            img.draft("L", (QUALITY_ANALYSIS_SIZE * 2,) * 2)  # JPEG fast path
        except Exception:
            pass

    if orientation in (5, 6, 7, 8):
        width, height = height, width

    gray = img.convert("L")
    gray.thumbnail((QUALITY_ANALYSIS_SIZE, QUALITY_ANALYSIS_SIZE), Image.BILINEAR)
    return gray, int(width), int(height)


def _laplacian_variance(gray):
    """Variance of a 4-neighbour Laplacian on the grayscale thumbnail."""
    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0
    lap = (arr[:-2, 1:-1] + arr[2:, 1:-1]
           + arr[1:-1, :-2] + arr[1:-1, 2:]) - 4.0 * arr[1:-1, 1:-1]
    return float(lap.var())


def _lighting(arr):
    mean = float(arr.mean())
    std = float(arr.std())
    hist = np.bincount(arr.astype(np.uint8).ravel(), minlength=256)
    total = float(arr.size) or 1.0
    shadow = float(hist[:16].sum()) / total
    highlight = float(hist[240:].sum()) / total
    return mean, std, shadow, highlight


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def analyze_image_quality(source):
    """Fast pre-inference quality report.

    ``source`` may be raw image bytes or a PIL image. Returns::

        {
          "ok": bool, "score": 0-100, "action": "proceed" | "adjust" | "retake",
          "issues": ["blurry", "too_dark", ...],
          "is_blurry": bool, "sharpness_score": float, "blur_threshold": float,
          "lighting_issue": None | "too_dark" | "overexposed",
          "mean_luminance": float, "contrast": float,
          "shadow_clip_pct": float, "highlight_clip_pct": float,
          "width": int, "height": int, "min_dimension": int,
          "aspect_ratio": float, "resolution_issue": bool,
          "aspect_issue": bool,
          "tips": [{"en": str, "ta": str, "hi": str}, ...],
          "tips_by_language": {"en": [...], "ta": [...], "hi": [...]},
          "summary": {"en": str, "ta": str, "hi": str},
          "latency_ms": float
        }
    """
    started = time.perf_counter()

    if not QUALITY_ENABLED:
        return _empty_report(started, "disabled")

    try:
        gray, width, height = _to_gray(source)
    except Exception as exc:
        return _unreadable_report(started, exc)

    arr = np.asarray(gray, dtype=np.float32)
    sharpness = round(_laplacian_variance(gray), 2)
    mean, std, shadow, highlight = _lighting(arr)

    min_dim = min(width, height)
    long_side, short_side = max(width, height), max(1, min(width, height))
    aspect = long_side / short_side

    is_blurry = sharpness < QUALITY_BLUR_THRESHOLD
    if mean < QUALITY_DARK_MEAN:
        lighting_issue = "too_dark"
    elif mean > QUALITY_BRIGHT_MEAN:
        lighting_issue = "overexposed"
    else:
        lighting_issue = None

    resolution_issue = min_dim < QUALITY_MIN_DIM
    aspect_issue = aspect > QUALITY_MAX_ASPECT
    clipping_issue = shadow >= QUALITY_SHADOW_CLIP or highlight >= QUALITY_HIGHLIGHT_CLIP

    issues = []
    if is_blurry:
        issues.append("blurry")
    if lighting_issue:
        issues.append(lighting_issue)
    if resolution_issue:
        issues.append("low_resolution")
    if aspect_issue:
        issues.append("extreme_aspect")

    tips = [_tip(key) for key in ("blurry" if is_blurry else None,
                                  lighting_issue,
                                  "low_resolution" if resolution_issue else None,
                                  "extreme_aspect" if aspect_issue else None)
            if key]

    score = 100
    if is_blurry:
        score -= 35
    if lighting_issue:
        score -= 25
    if resolution_issue:
        score -= 20
    if aspect_issue:
        score -= 10
    if clipping_issue and not lighting_issue:
        score -= 5
    score = max(0, min(100, score))

    if is_blurry or lighting_issue or resolution_issue:
        action = "retake"
    elif aspect_issue or clipping_issue:
        action = "adjust"
    else:
        action = "proceed"

    report = {
        "ok": not issues,
        "score": score,
        "action": action,
        "issues": issues,
        "is_blurry": is_blurry,
        "sharpness_score": sharpness,
        "blur_threshold": QUALITY_BLUR_THRESHOLD,
        "lighting_issue": lighting_issue,
        "mean_luminance": round(mean, 1),
        "contrast": round(std, 1),
        "shadow_clip_pct": round(shadow * 100, 2),
        "highlight_clip_pct": round(highlight * 100, 2),
        "clipping_issue": clipping_issue,
        "width": width,
        "height": height,
        "min_dimension": min_dim,
        "aspect_ratio": round(aspect, 3),
        "resolution_issue": resolution_issue,
        "aspect_issue": aspect_issue,
        "tips": tips,
        "tips_by_language": {lang: [tip[lang] for tip in tips] for lang in LANGS},
        "summary": _summary(issues),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    return report


def _summary(issues):
    """One-line localized summary: the tips for every distinct issue."""
    if not issues:
        return {lang: TIPS["good_angle"][lang] for lang in LANGS}
    return {
        lang: " ".join(TIPS[key][lang] for key in issues if key in TIPS)
        for lang in LANGS
    }


def _empty_report(started, reason):
    return {
        "ok": True, "score": 100, "action": "proceed", "issues": [],
        "is_blurry": False, "sharpness_score": 0.0,
        "blur_threshold": QUALITY_BLUR_THRESHOLD, "lighting_issue": None,
        "mean_luminance": None, "contrast": None,
        "shadow_clip_pct": None, "highlight_clip_pct": None,
        "clipping_issue": False, "width": None, "height": None,
        "min_dimension": None, "aspect_ratio": None,
        "resolution_issue": False, "aspect_issue": False,
        "tips": [], "tips_by_language": {"en": [], "ta": [], "hi": []},
        "summary": {lang: "" for lang in LANGS},
        "disabled": True, "reason": reason,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _unreadable_report(started, exc):
    tip = _tip("unreadable")
    return {
        "ok": False, "score": 0, "action": "retake", "issues": ["unreadable"],
        "is_blurry": False, "sharpness_score": 0.0,
        "blur_threshold": QUALITY_BLUR_THRESHOLD, "lighting_issue": None,
        "mean_luminance": None, "contrast": None,
        "shadow_clip_pct": None, "highlight_clip_pct": None,
        "clipping_issue": False, "width": None, "height": None,
        "min_dimension": None, "aspect_ratio": None,
        "resolution_issue": False, "aspect_issue": False,
        "tips": [tip], "tips_by_language": {lang: [tip[lang]] for lang in LANGS},
        "summary": {lang: tip[lang] for lang in LANGS},
        "error": f"{type(exc).__name__}: {exc}",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


# ---------------------------------------------------------------------------
#  OOD explainer
# ---------------------------------------------------------------------------

def explain_ood_for_user(is_ood, ood_details=None, quality_report=None):
    """Translate OOD internals into farmer-facing guidance.

    Two failure modes matter in the field:
      * the species head is not confident -> likely not an animal at all;
      * the animal is there but the breed evidence is weak -> a better side
        profile usually fixes it.

    Raw energy/MSP values are kept under ``technical`` for developer views;
    the headline stays plain-language in English, Tamil and Hindi.
    """
    details = dict(ood_details or {})
    quality = quality_report or {}
    quality_tips = list(quality.get("tips") or [])

    msp = _as_float(details.get("msp"))
    floor = _as_float(details.get("binary_confidence_floor"))
    energy = _as_float(details.get("energy"))
    energy_threshold = _as_float(details.get("energy_threshold"))
    msp_threshold = _as_float(details.get("msp_threshold"))

    technical = {
        "msp": msp,
        "msp_threshold": msp_threshold,
        "binary_confidence_floor": floor,
        "energy": energy,
        "energy_threshold": energy_threshold,
        "ood_by_binary_floor": bool(details.get("ood_by_binary_floor")),
        "ood_by_msp": bool(details.get("ood_by_msp")),
        "ood_by_energy": bool(details.get("ood_by_energy")),
    }

    if not is_ood:
        code = "poor_quality" if quality and quality.get("ok") is False else "ok"
        severity = "warning" if code == "poor_quality" else "ok"
    elif floor is not None and msp is not None and msp < floor:
        code = "not_animal"
        severity = "error"
    elif ((energy is not None and energy_threshold is not None
           and energy > energy_threshold)
          or (msp is not None and msp_threshold is not None
              and msp < msp_threshold)):
        code = "unclear_features"
        severity = "warning"
    else:
        # OOD was flagged but no numeric reason was forwarded (e.g. a custom
        # detector). Asking for a better photo is the safe, non-alarming answer.
        code = "unclear_features"
        severity = "warning"

    block = MESSAGES[code]
    # Deduplicate while preserving order (quality first, then the OOD cue).
    tips = []
    seen = set()
    for tip in quality_tips + [_tip(k) for k in block["tips"]]:
        marker = tip["en"]
        if marker in seen:
            continue
        seen.add(marker)
        tips.append(tip)

    return {
        "is_ood": bool(is_ood),
        "code": code,
        "severity": severity,
        "title": dict(block["title"]),
        "message": dict(block["message"]),
        "tips": tips,
        "tips_by_language": {lang: [t[lang] for t in tips] for lang in LANGS},
        "quality_score": quality.get("score"),
        "technical": technical,
    }


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
