"""Lightweight, model-specific inference used by the Developer Console.

Unlike ``server.services.inference.predict_image`` this helper runs **only**
the selected model (no CLIP attributes, advisory or knowledge lookups), which
makes it appropriate for comparing checkpoints/ONNX exports in the dev UI.
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F
from PIL import Image

from src.image_quality import analyze_image_quality


def _pick_output(bundle, out_dict, ort_outs, kind, fallback):
    name = (bundle.spec or {}).get("onnx_outputs", {}).get(kind)
    if name and name in out_dict:
        return out_dict[name]
    return out_dict.get(kind, ort_outs[fallback])


def _forward(bundle, tensor):
    if bundle.is_onnx:
        input_name = bundle.onnx_input_name
        if not input_name:
            input_name = bundle.model_or_session.get_inputs()[0].name
        arr = tensor.numpy()
        if "float16" in str(bundle.input_dtype).lower():
            arr = arr.astype("float16")
        outs = bundle.model_or_session.run(None, {input_name: arr})
        names = [o.name for o in bundle.model_or_session.get_outputs()]
        out_dict = dict(zip(names, outs))
        binary = torch.tensor(_pick_output(bundle, out_dict, outs, "binary", 0)[0]).float()
        cattle = torch.tensor(_pick_output(bundle, out_dict, outs, "cattle", 1)[0]).float()
        buffalo = torch.tensor(_pick_output(bundle, out_dict, outs, "buffalo", 2)[0]).float()
        return binary, cattle, buffalo

    model = bundle.model_or_session
    first = next(model.parameters(), None)
    x = tensor.half() if first is not None and first.dtype == torch.float16 else tensor.float()
    try:
        x = x.to(next(model.parameters()).device)
    except StopIteration:
        pass
    with torch.no_grad():
        out = model(x)
    return out["binary"][0].float(), out["cattle"][0].float(), out["buffalo"][0].float()


def run_model_test(manager, model_name: str, image_bytes: bytes,
                   repeats: int = 1) -> Dict[str, Any]:
    """Run the selected model on one image and return a compact report."""
    if model_name not in manager.models:
        raise ValueError(f"Model '{model_name}' not found")

    bundle = manager._load_bundle(model_name)
    repeats = max(1, min(int(repeats or 1), 10))

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Invalid image: {exc}") from exc

    quality = analyze_image_quality(img)
    tensor = bundle.transform(img).unsqueeze(0)

    latencies = []
    binary = cattle = buffalo = None
    for _ in range(repeats):
        start = time.perf_counter()
        binary, cattle, buffalo = _forward(bundle, tensor)
        latencies.append((time.perf_counter() - start) * 1000.0)

    binary_probs = F.softmax(binary, dim=0)
    cattle_probs = F.softmax(cattle, dim=0)
    buffalo_probs = F.softmax(buffalo, dim=0)
    combined = torch.cat([binary_probs[0] * cattle_probs,
                          binary_probs[1] * buffalo_probs])

    n_cattle = cattle.numel()
    top_probs, top_idxs = combined.topk(min(5, combined.numel()))
    top5 = []
    for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        if idx < n_cattle:
            slug = bundle.cattle_inv_map.get(idx, f"cattle_{idx}")
        else:
            slug = bundle.buffalo_inv_map.get(idx - n_cattle, f"buffalo_{idx - n_cattle}")
        top5.append({
            "breed": slug.replace("_", " ").title(),
            "breed_slug": slug,
            "confidence": round(prob * 100, 2),
        })

    species_idx = int(torch.argmax(binary_probs).item())
    detector = getattr(manager, "ood_detector", None)
    if detector is not None:
        is_ood, ood_score, ood_details = detector.is_ood(
            binary.unsqueeze(0), cattle.unsqueeze(0), buffalo.unsqueeze(0))
    else:
        is_ood, ood_score, ood_details = False, 0.0, {"enabled": False}

    return {
        "model": model_name,
        "engine": bundle.engine,
        "species": "Buffalo" if species_idx == 1 else "Cattle",
        "species_confidence": round(binary_probs[species_idx].item() * 100, 2),
        "binary_probs": {
            "cattle": round(binary_probs[0].item(), 6),
            "buffalo": round(binary_probs[1].item(), 6),
        },
        "breed": top5[0]["breed"] if top5 else "Unknown",
        "breed_slug": top5[0]["breed_slug"] if top5 else None,
        "confidence": top5[0]["confidence"] if top5 else 0.0,
        "top5": top5,
        "is_ood": bool(is_ood),
        "ood_score": round(float(ood_score), 4),
        "ood_details": ood_details,
        "quality": quality,
        "preprocessing": {
            "image_size": bundle.image_size,
            "normalization_baked": bundle.normalization_baked,
            "input_dtype": bundle.input_dtype,
        },
        "class_counts": {"cattle": bundle.num_cattle, "buffalo": bundle.num_buffalo},
        "latency_ms": {
            "last": round(latencies[-1], 2),
            "mean": round(sum(latencies) / len(latencies), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
            "runs": len(latencies),
        },
    }


__all__ = ["run_model_test"]
