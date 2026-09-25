import os
import json
from flask import Blueprint, jsonify, request
from server.model_manager import get_manager
from src.config import CHECKPOINT_DIR, IMAGE_SIZE

models_bp = Blueprint('models', __name__)

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
    layer_summary = []

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

    spec["architecture"] = {
        "feature_dim": 1280,
        "binary_head": "Linear(1280→256→2)",
        "cattle_head": "Linear(1280→512→57) + Dropout(0.4)",
        "buffalo_head": "Linear(1280→512→18) + Dropout(0.4)",
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
    return jsonify({"models": models, "device": device})

@models_bp.route('/api/model-spec', methods=['GET'])
def model_spec():
    manager = get_manager()
    model_name = request.args.get("model", "")
    if model_name not in manager.models:
        return jsonify({"error": "Model not found"}), 404
    try:
        model = manager._load_model(model_name)
        spec = get_model_spec(model, model_name, manager.models[model_name])
        return jsonify(spec)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@models_bp.route('/api/upload-model', methods=['POST'])
def upload_model():
    if 'model_file' not in request.files:
        return jsonify({"error": "No model file provided"}), 400
    
    file = request.files['model_file']
    filename = request.form.get("filename", "uploaded_model.pt")
    
    upload_dir = os.path.join(CHECKPOINT_DIR, "uploaded_models")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, filename)
    file.save(save_path)
    
    get_manager()._discover_models()
    return jsonify({"status": "success"})
