import time
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.config import EVAL_MATCH_TRAIN_RESOLUTION, TRAIN_RESIZE, IMAGE_SIZE, SPECIES_LABELS
from server.utils.logger import get_logger
from server.shared_config import get_shared_config

_EVAL_RESIZE = TRAIN_RESIZE if EVAL_MATCH_TRAIN_RESOLUTION else IMAGE_SIZE
TRANSFORM = transforms.Compose([
    transforms.Resize(_EVAL_RESIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Five-level confidence rubric (report Module 6)
# ---------------------------------------------------------------------------
CONFIDENCE_BANDS = (
    (0.85, 5, "very_high"),
    (0.75, 4, "high"),
    (0.60, 3, "moderate"),
    (0.40, 2, "low"),
    (0.00, 1, "very_low"),
)


def confidence_level(confidence, is_ood=False):
    """Map joint soft-routed confidence C=P(species)*P(breed|species) to 1..5."""
    if is_ood:
        return 1
    c = float(confidence or 0.0)
    for threshold, level, _band in CONFIDENCE_BANDS:
        if c >= threshold:
            return level
    return 1


def confidence_band(level):
    for _threshold, candidate, band in CONFIDENCE_BANDS:
        if candidate == level:
            return band
    return "very_low"


# ---------------------------------------------------------------------------
# Explainability: CBAM spatial attention + class-activation heatmap
# ---------------------------------------------------------------------------

def _install_xray_capture(model):
    """Attach non-destructive hooks before a forward pass.

    ``avg_pool`` receives the final convolutional feature map F ([1,C,H,W]),
    and ``attention.spatial_attention`` exposes the CBAM 7x7 spatial gate.
    Every hook is removed immediately after the forward pass.
    """
    captured = {}
    handles = []

    avg_pool = getattr(model, "avg_pool", None)
    if avg_pool is not None:
        def _pool_hook(_module, inputs):
            if inputs:
                captured["features"] = inputs[0].detach()
        handles.append(avg_pool.register_forward_pre_hook(_pool_hook))

    attention = getattr(model, "attention", None)
    spatial = getattr(attention, "spatial_attention", None)
    if spatial is not None:
        def _spatial_hook(module, inputs, _output):
            try:
                x = inputs[0]
                with torch.no_grad():
                    avg = torch.mean(x, dim=1, keepdim=True)
                    mx = torch.amax(x, dim=1, keepdim=True)
                    att = module.conv(torch.cat([avg, mx], dim=1))
                    if getattr(module, "identity_init", False):
                        mask = 1.0 + (2.0 * torch.sigmoid(att) - 1.0)
                    else:
                        mask = torch.sigmoid(att)
                    captured["spatial_mask"] = mask.detach()
            except Exception:
                pass
        handles.append(spatial.register_forward_hook(_spatial_hook))

    return captured, handles


def _remove_hooks(handles):
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass


def _normalize_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    arr = np.maximum(arr, 0.0)
    lo, hi = np.percentile(arr, [2.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _colorize(norm, alpha=0.86):
    """Turbo/Jet RGBA heatmap; alpha is proportional to activation."""
    try:
        from matplotlib import colormaps
        rgba = colormaps["turbo"](norm)
    except Exception:
        # Minimal turbo-like fallback (blue → cyan → green → yellow → red).
        stops = np.array([
            [0.19, 0.07, 0.23], [0.10, 0.50, 0.85], [0.12, 0.83, 0.55],
            [0.95, 0.86, 0.20], [0.85, 0.15, 0.10],
        ], dtype=np.float32)
        pos = np.linspace(0, 1, len(stops))
        rgba = np.zeros(norm.shape + (4,), dtype=np.float32)
        for c in range(3):
            rgba[..., c] = np.interp(norm, pos, stops[:, c])
        rgba[..., 3] = 1.0
    rgba = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
    fade = np.clip((norm - 0.06) / 0.94, 0, 1) ** 1.15
    rgba[..., 3] = (fade * 255 * alpha).astype(np.uint8)
    return rgba


def _rgba_data_url(rgba):
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _quadrant_metadata(norm):
    h, w = norm.shape
    half_h, half_w = h // 2, w // 2
    blocks = {
        "top_left": float(norm[:half_h, :half_w].mean()),
        "top_right": float(norm[:half_h, half_w:].mean()),
        "bottom_left": float(norm[half_h:, :half_w].mean()),
        "bottom_right": float(norm[half_h:, half_w:].mean()),
    }
    total = sum(blocks.values()) or 1.0
    peak_y, peak_x = np.unravel_index(int(np.argmax(norm)), norm.shape)
    return {
        "dominant_quadrant": max(blocks, key=blocks.get),
        "quadrant_scores": {k: round(v / total, 4) for k, v in blocks.items()},
        "peak_px": [int(peak_x), int(peak_y)],
    }


def _build_xray_payload(model, captured, combined, n_cattle, size=IMAGE_SIZE):
    """Project the winning head's linear weights onto F and encode heatmaps."""
    feats = captured.get("features")
    mask = captured.get("spatial_mask")
    if feats is None:
        return {
            "attention_map_b64": None,
            "spatial_attention_b64": None,
            "attention_meta": {"source": "unavailable"},
        }

    try:
        top_idx = int(torch.argmax(combined).item())
        head = model.cattle_head if top_idx < n_cattle else model.buffalo_head
        class_idx = top_idx if top_idx < n_cattle else top_idx - n_cattle
        last = head[-1]
        weights = getattr(last, "weight", None)
        fmap = feats[0].float()  # [C,H,W]
        if weights is not None and weights.shape[-1] == fmap.shape[0]:
            cam = torch.einsum("c,chw->hw", weights[class_idx].float(), fmap)
        else:
            # Fallback: L2 activation norm across channels.
            cam = torch.linalg.vector_norm(fmap, dim=0)
        cam = torch.relu(cam)
        cam = F.interpolate(
            cam[None, None], size=(size, size), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        cam_norm = _normalize_map(cam)

        spatial_norm = None
        if mask is not None:
            spatial = F.interpolate(
                mask.float(), size=(size, size), mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()
            spatial_norm = _normalize_map(spatial)

        cam_url = _rgba_data_url(_colorize(cam_norm))
        spatial_url = _rgba_data_url(_colorize(spatial_norm, alpha=0.72)) if spatial_norm is not None else None

        meta = {
            "source": "cbam_spatial+linear_cam" if spatial_norm is not None else "linear_cam",
            "feature_grid": [int(feats.shape[-2]), int(feats.shape[-1])],
            "cbam_grid": [int(mask.shape[-2]), int(mask.shape[-1])] if mask is not None else None,
            "winning_class_index": class_idx,
            "winning_species": "cattle" if top_idx < n_cattle else "buffalo",
            **_quadrant_metadata(cam_norm),
        }
        return {
            "attention_map_b64": cam_url,
            "spatial_attention_b64": spatial_url,
            "attention_meta": meta,
        }
    except Exception as exc:
        logger = get_logger()
        if logger:
            logger.warning(f"Attention extraction skipped: {exc}")
        return {
            "attention_map_b64": None,
            "spatial_attention_b64": None,
            "attention_meta": {"source": "error", "error": str(exc)},
        }


def predict_image(manager, image_bytes, model_name=None, socketio=None, filename="unknown"):
    """Run prediction on raw image bytes. Returns result dict."""
    logger = get_logger()
    predict_start = time.time()

    if not model_name:
        model_name = manager.get_active_model_name()

    if not model_name or model_name not in manager.models:
        if logger:
            logger.error(f"Model '{model_name}' not found")
        return {"error": f"Model '{model_name}' not found"}

    if logger:
        logger.info(f"Image received ({len(image_bytes) / 1024:.1f} KB), filename='{filename}', model='{model_name}'")

    if socketio:
        socketio.emit('prediction_progress', {'stage': 'preprocessing', 'detail': 'Decoding image'}, namespace='/ws')

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        if logger:
            logger.error(f"Invalid image: {e}")
        return {"error": f"Invalid image: {e}"}

    if socketio:
        socketio.emit('prediction_progress', {
            'stage': 'preprocessing', 
            'detail': f'Resizing {img.width}x{img.height} to {IMAGE_SIZE}x{IMAGE_SIZE}'
        }, namespace='/ws')

    tensor = TRANSFORM(img).unsqueeze(0)
    info = manager.models[model_name]

    try:
        model = manager._load_model(model_name, socketio=socketio)
    except Exception as e:
        if logger:
            logger.error(f"Model load failed: {e}")
        return {"error": str(e)}

    inference_start = time.time()

    if socketio:
        socketio.emit('prediction_progress', {'stage': 'inference', 'detail': f'Running model on {manager.device.type}'}, namespace='/ws')

    if info.get("type") == "onnx":
        input_name = model.get_inputs()[0].name
        ort_outs = model.run(None, {input_name: tensor.numpy()})
        out_names = [x.name for x in model.get_outputs()]
        out_dict = dict(zip(out_names, ort_outs))
        binary_logits = torch.tensor(out_dict.get("binary", ort_outs[0])[0])
        cattle_logits = torch.tensor(out_dict.get("cattle", ort_outs[1])[0])
        buffalo_logits = torch.tensor(out_dict.get("buffalo", ort_outs[2])[0])
        xray_captured = {}
    else:
        xray_captured, xray_handles = _install_xray_capture(model)
        try:
            with torch.no_grad():
                first_param = next(model.parameters(), None)
                if first_param is not None and first_param.dtype == torch.float16:
                    tensor = tensor.half()
                else:
                    tensor = tensor.float()

                tensor = tensor.to(manager.device)
                out = model(tensor)
        finally:
            _remove_hooks(xray_handles)

        if isinstance(out, (tuple, list)):
            binary_logits = out[0][0].float()
            cattle_logits = out[1][0].float()
            buffalo_logits = out[2][0].float()
        elif isinstance(out, dict):
            binary_logits = out["binary"][0].float()
            cattle_logits = out["cattle"][0].float()
            buffalo_logits = out["buffalo"][0].float()
        else:
            raise ValueError(f"Unexpected model output structure: {type(out)}")

    inference_time = time.time() - inference_start

    if socketio:
        socketio.emit('prediction_progress', {'stage': 'postprocessing', 'detail': 'Routing soft predictions'}, namespace='/ws')

    is_ood = False
    ood_score = 0.0
    ood_details = {}
    db = get_shared_config(socketio)
    ood_active = db.get("ood.enabled", True)
    if ood_active and manager.ood_detector is not None:
        is_ood, ood_score, ood_details = manager.ood_detector.is_ood(
            binary_logits.unsqueeze(0), cattle_logits.unsqueeze(0), buffalo_logits.unsqueeze(0)
        )

    binary_probs = F.softmax(binary_logits, dim=0)
    cattle_probs = F.softmax(cattle_logits, dim=0)
    buffalo_probs = F.softmax(buffalo_logits, dim=0)
    cattle_map = manager.class_maps.get("cattle", {})
    buffalo_map = manager.class_maps.get("buffalo", {})
    cattle_inv = {v: k for k, v in cattle_map.items()}
    buffalo_inv = {v: k for k, v in buffalo_map.items()}
    n_cattle = cattle_logits.numel()
    combined = torch.cat([binary_probs[0] * cattle_probs,
                          binary_probs[1] * buffalo_probs])

    species_map_inv = {v: k for k, v in SPECIES_LABELS.items()}
    primary_species_idx = int(torch.argmax(binary_probs).item())
    species_prob = round(binary_probs[primary_species_idx].item(), 4)

    if is_ood:
        top5 = [{"breed": "Not a recognized animal", "confidence": 0.0}]
        species_name = "Unknown"
        species_conf = 0.0
    else:
        # Retrieve configurable breed display limits
        min_c = float(db.get("display.min_breed_confidence_pct", 5.0))
        min_b = int(db.get("display.min_breeds", 1))
        max_b = int(db.get("display.max_breeds", db.get("display.top_k", 5)))
        fetch_k = min(max(max_b, 5), combined.numel())

        top_probs, top_idxs = combined.topk(fetch_k)
        candidates = []
        for prob, idx in zip(top_probs.tolist(), top_idxs.tolist()):
            if idx < n_cattle:
                breed = cattle_inv.get(idx, f"cattle_{idx}")
            else:
                breed = buffalo_inv.get(idx - n_cattle, f"buffalo_{idx - n_cattle}")
            display_name = breed.replace("_", " ").title()
            candidates.append({
                "breed": display_name,
                "confidence": round(prob * 100, 2),
            })

        # Apply minimum confidence cutoff while guaranteeing at least min_b breeds
        filtered = [c for c in candidates if c["confidence"] >= min_c]
        if len(filtered) < min_b:
            filtered = candidates[:min_b]
        top5 = filtered[:max_b]

        species_idx = 0 if int(top_idxs[0]) < n_cattle else 1
        species_name = species_map_inv.get(species_idx, "cattle").title()
        species_conf = binary_probs[species_idx].item() * 100

    total_time = time.time() - predict_start

    top_breed_name = top5[0]["breed"] if top5 else "Unknown"
    top1_combined = float(combined.max().item()) if not is_ood else 0.0
    top_breed_conf = top1_combined
    level = confidence_level(top_breed_conf, is_ood=is_ood)

    xray = _build_xray_payload(model, xray_captured, combined, n_cattle) \
        if info.get("type") != "onnx" else {
            "attention_map_b64": None,
            "spatial_attention_b64": None,
            "attention_meta": {"source": "onnx_unavailable"},
        }

    result = {
        "species": species_name,
        "predicted_species": species_name,
        "species_confidence": round(species_conf, 2),
        "species_prob": species_prob,
        "top_breed": top_breed_name,
        "predicted_breed": top_breed_name,
        "confidence": top_breed_conf,
        "top_breed_confidence": round(top_breed_conf * 100, 2),
        "confidence_level": level,
        "confidence_band": confidence_band(level),
        "confidence_action": f"confidence.level{level}.action",
        "top5_breeds": top5,
        "top_k": [{"breed": b["breed"], "prob": b["confidence"] / 100.0} for b in top5],
        "model_used": model_name,
        "model_name": model_name,
        "inference_time_ms": round(inference_time * 1000, 1),
        "total_time_ms": round(total_time * 1000, 1),
        "routing": "soft",
        "ood": is_ood,
        "is_ood": is_ood,
        "ood_score": round(ood_score, 2) if is_ood else None,
        "ood_details": ood_details,
        "attention_map_b64": xray.get("attention_map_b64"),
        "spatial_attention_b64": xray.get("spatial_attention_b64"),
        "attention_meta": xray.get("attention_meta"),
    }

    result["_raw_binary_probs"] = [round(p, 6) for p in binary_probs.tolist()]
    k10 = min(10, combined.numel())
    raw_probs, raw_idxs = combined.topk(k10)
    result["_raw_breed_probs_top10"] = [
        {
            "idx": int(idx),
            "species": "cattle" if int(idx) < n_cattle else "buffalo",
            "breed": (cattle_inv.get(int(idx), f"cattle_{int(idx)}")
                      if int(idx) < n_cattle
                      else buffalo_inv.get(int(idx) - n_cattle,
                                           f"buffalo_{int(idx) - n_cattle}")),
            "prob": round(float(p), 6),
        }
        for p, idx in zip(raw_probs.tolist(), raw_idxs.tolist())
    ]

    try:
        from src.run_utils import log_event
        log_event("prediction", category="test", model=model_name,
                  filename=filename,
                  species=result.get("species"),
                  species_confidence=result.get("species_confidence"),
                  top_breed=result.get("top_breed"),
                  top_breed_confidence=result.get("top_breed_confidence"),
                  top5=[b["breed"] for b in result.get("top5_breeds", [])],
                  latency_ms=result.get("inference_time_ms"),
                  routing=result.get("routing"))
    except Exception:
        pass

    if socketio:
        socketio.emit('prediction_complete', {'result': result}, namespace='/ws')

    return result
