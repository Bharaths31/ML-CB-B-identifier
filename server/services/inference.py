import time
import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.config import EVAL_MATCH_TRAIN_RESOLUTION, TRAIN_RESIZE, IMAGE_SIZE, SPECIES_LABELS
from server.utils.logger import get_logger

_EVAL_RESIZE = TRAIN_RESIZE if EVAL_MATCH_TRAIN_RESOLUTION else IMAGE_SIZE
TRANSFORM = transforms.Compose([
    transforms.Resize(_EVAL_RESIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_image(manager, image_bytes, model_name, socketio=None):
    """Run prediction on raw image bytes. Returns result dict."""
    logger = get_logger()
    predict_start = time.time()

    if model_name not in manager.models:
        if logger:
            logger.error(f"Model '{model_name}' not found")
        return {"error": f"Model '{model_name}' not found"}

    if logger:
        logger.info(f"Image received ({len(image_bytes) / 1024:.1f} KB), model='{model_name}'")

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
    else:
        with torch.no_grad():
            tensor = tensor.to(manager.device)
            out = model(tensor)
            binary_logits = out["binary"][0]
            cattle_logits = out["cattle"][0]
            buffalo_logits = out["buffalo"][0]

    inference_time = time.time() - inference_start

    if socketio:
        socketio.emit('prediction_progress', {'stage': 'postprocessing', 'detail': 'Routing soft predictions'}, namespace='/ws')

    is_ood = False
    ood_score = 0.0
    ood_details = {}
    if manager.ood_detector is not None:
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

    if is_ood:
        top5 = [{"breed": "Not a recognized animal", "confidence": 0.0}]
        species_name = "Unknown"
        species_conf = 0.0
    else:
        k = min(5, combined.numel())
        top5_probs, top5_idxs = combined.topk(k)
        top5 = []
        for prob, idx in zip(top5_probs.tolist(), top5_idxs.tolist()):
            if idx < n_cattle:
                breed = cattle_inv.get(idx, f"cattle_{idx}")
            else:
                breed = buffalo_inv.get(idx - n_cattle, f"buffalo_{idx - n_cattle}")
            display_name = breed.replace("_", " ").title()
            top5.append({
                "breed": display_name,
                "confidence": round(prob * 100, 2),
            })

        species_idx = 0 if int(top5_idxs[0]) < n_cattle else 1
        species_name = SPECIES_LABELS[species_idx]
        species_conf = binary_probs[species_idx].item() * 100

    total_time = time.time() - predict_start

    result = {
        "species": species_name,
        "species_confidence": round(species_conf, 2),
        "top_breed": top5[0]["breed"] if top5 else "Unknown",
        "top_breed_confidence": top5[0]["confidence"] if top5 else 0,
        "top5_breeds": top5,
        "model_used": model_name,
        "inference_time_ms": round(inference_time * 1000, 1),
        "total_time_ms": round(total_time * 1000, 1),
        "routing": "soft",
        "ood": is_ood,
        "ood_score": round(ood_score, 2) if is_ood else None,
        "ood_details": ood_details
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
        from src.run_logger import log_event
        log_event("prediction", category="test", model=model_name,
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
