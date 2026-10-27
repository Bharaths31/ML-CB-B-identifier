import os
import json
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
from server.model_manager import get_manager, head_out_features
from server.utils.logger import get_logger
from src.config import CHECKPOINT_DIR, IMAGE_SIZE

models_bp = Blueprint('models', __name__)

ALLOWED_MODEL_EXTENSIONS = {".pt", ".onnx"}


def _head_dims(model, manager, name):
    """Return dynamic (cattle, buffalo) output widths for a loaded model."""
    if isinstance(model, dict):
        return model.get("cattle"), model.get("buffalo")
    counts = manager.class_counts(name)
    cattle = counts.get("cattle")
    buffalo = counts.get("buffalo")
    if hasattr(model, "cattle_head"):
        cattle = head_out_features(model.cattle_head) or cattle
        buffalo = head_out_features(model.buffalo_head) or buffalo
    elif hasattr(model, "get_outputs"):
        try:
            for output in model.get_outputs():
                width = next((v for v in reversed(output.shape or []) if isinstance(v, int) and v > 0), None)
                if output.name == "cattle":
                    cattle = width or cattle
                elif output.name == "buffalo":
                    buffalo = width or buffalo
        except Exception:
            pass
    return cattle, buffalo


def get_model_spec(model, name, info):
    """Extract detailed model architecture and parameter info."""
    manager = get_manager()
    cattle_count, buffalo_count = _head_dims(model, manager, name)
    spec = {
        "name": name,
        "backbone": info.get("backbone", "unknown"),
        "type": info.get("type", "pt"),
        "path": info.get("path", ""),
        "device": str(next(model.parameters()).device) if hasattr(model, "parameters") else "N/A",
        "image_size": IMAGE_SIZE,
        "class_counts": {
            "cattle": cattle_count,
            "buffalo": buffalo_count,
            "total": (cattle_count or 0) + (buffalo_count or 0),
        },
        "class_map_source": manager.class_map_source,
        "class_map_counts": {
            "cattle": len(manager.class_maps.get("cattle", {})),
            "buffalo": len(manager.class_maps.get("buffalo", {})),
        },
    }

    if info.get("type") == "onnx":
        try:
            spec["inputs"] = [{"name": i.name, "shape": str(i.shape), "type": i.type}
                              for i in model.get_inputs()]
            spec["outputs"] = [{"name": o.name, "shape": str(o.shape), "type": o.type}
                               for o in model.get_outputs()]
        except Exception:
            pass
        return spec

    total_params = 0
    trainable_params = 0

    for pname, param in model.named_parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count

    spec["total_parameters"] = total_params
    spec["trainable_parameters"] = trainable_params
    spec["frozen_parameters"] = total_params - trainable_params

    def _human_number(n):
        if n >= 1_000_000: return f"{n / 1_000_000:.2f}M"
        if n >= 1_000: return f"{n / 1_000:.1f}K"
        return str(n)

    spec["total_parameters_human"] = _human_number(total_params)
    spec["model_size_mb"] = round(total_params * 4 / (1024 * 1024), 2)

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

    binary_dim = None
    breed_dim = None
    try:
        binary_dim = model.binary_head[0].out_features
    except Exception:
        pass
    try:
        breed_dim = model.cattle_head[0].out_features
    except Exception:
        pass
    spec["head_dimensions"] = {
        "feature_dim": int(getattr(model, "features_dim", 1280) or 1280),
        "binary_width": binary_dim,
        "breed_width": breed_dim,
        "cattle_classes": cattle_count,
        "buffalo_classes": buffalo_count,
    }
    spec["architecture"] = {
        "feature_dim": 1280,
        "binary_head": f"Linear(1280→{binary_dim or '?'}×2) → 2",
        "cattle_head": f"Linear(1280→{breed_dim or '?'}×2) → {cattle_count or '?'} cattle classes",
        "buffalo_head": f"Linear(1280→{breed_dim or '?'}×2) → {buffalo_count or '?'} buffalo classes",
        "attention": "CBAM (after stage 3)",
        "pooling": "AdaptiveAvgPool2d(1)",
    }

    model_info_path = os.path.join(os.path.dirname(info.get("path", "")), "model_info.json")
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path) as f:
                spec["model_info_json"] = json.load(f)
        except Exception:
            pass

    return spec

@models_bp.route('/api/models', methods=['GET'])
def list_models():
    manager = get_manager()
    models = manager.list_models()
    device = "cuda" if manager.device.type == "cuda" else "cpu"
    active_model = manager.get_active_model_name()
    return jsonify({"models": models, "device": device, "active_model": active_model})

@models_bp.route('/api/select-model', methods=['POST'])
def select_model():
    manager = get_manager()
    from flask import current_app
    socketio = current_app.extensions.get('socketio')
    data = request.get_json(silent=True) or {}
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "Missing 'model' field"}), 400
    try:
        manager.set_active_model(model_name, socketio=socketio)
        return jsonify({
            "status": "ok",
            "active_model": manager.get_active_model_name(),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger = get_logger()
        if logger:
            logger.error(f"Failed to switch model to '{model_name}': {e}")
        return jsonify({"error": f"Failed to switch model: {e}"}), 500

@models_bp.route('/api/model-spec', methods=['GET'])
def model_spec():
    manager = get_manager()
    model_name = request.args.get("model", "")
    if model_name not in manager.models:
        return jsonify({"error": "Model not found"}), 404
    try:
        model = manager._load_model(model_name)
        manager.bind_class_maps_for_model(model_name, model=model)
        spec = get_model_spec(model, model_name, manager.models[model_name])
        return jsonify(spec)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@models_bp.route('/api/upload-model', methods=['POST'])
def upload_model():
    if 'model_file' not in request.files:
        return jsonify({"error": "No model file provided"}), 400

    file = request.files['model_file']
    requested = request.form.get("filename") or file.filename or ""
    safe_name = secure_filename(requested)
    if not safe_name:
        return jsonify({"error": "Invalid filename"}), 400

    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_MODEL_EXTENSIONS:
        return jsonify({
            "error": "Only .pt and .onnx model files are allowed"
        }), 400
    if file.filename and os.path.splitext(file.filename)[1].lower() not in ALLOWED_MODEL_EXTENSIONS:
        return jsonify({
            "error": "Uploaded file extension must be .pt or .onnx"
        }), 400

    upload_dir = os.path.join(CHECKPOINT_DIR, "uploaded_models")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, safe_name)
    file.save(save_path)

    manager = get_manager()
    manager._discover_models()
    return jsonify({"status": "success", "filename": safe_name, "path": save_path})
