import os
import json
import shutil
import torch
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename
from server.model_manager import get_manager, head_out_features
from server.services.model_test import run_model_test
from server.utils.logger import get_logger
from src.config import CHECKPOINT_DIR, EXPORT_DIR, IMAGE_SIZE, SPLIT_DIR
from src.model_introspect import (build_model_metadata, embed_model_metadata,
                                  extract_state_dict, find_ood_json,
                                  inspect_and_load)
from src.run_utils import unique_path

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
    bundle = manager.bundles.get(name)
    image_size = bundle.image_size if bundle is not None else IMAGE_SIZE
    spec = {
        "name": name,
        "backbone": info.get("backbone", "unknown"),
        "type": info.get("type", "pt"),
        "path": info.get("path", ""),
        "device": str(next(model.parameters()).device) if hasattr(model, "parameters") else "N/A",
        "image_size": image_size,
        "self_describing": bool(getattr(bundle, "metadata", None)),
        "class_counts": {
            "cattle": cattle_count,
            "buffalo": buffalo_count,
            "total": (cattle_count or 0) + (buffalo_count or 0),
        },
        "class_map_source": (bundle.class_map_source if bundle is not None
                             else manager.class_map_source),
        "class_map_counts": {
            "cattle": len(manager.class_maps.get("cattle", {})),
            "buffalo": len(manager.class_maps.get("buffalo", {})),
        },
    }
    if bundle is not None:
        spec["embedded_metadata"] = bundle.metadata
        spec["preprocessing"] = bundle.spec_summary().get("preprocessing")
        spec["normalization_baked"] = bundle.normalization_baked
        spec["ood"] = _effective_ood(bundle)

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

    # Make the upload self-describing immediately: class maps, resolution,
    # normalization mode and OOD defaults are embedded in the file itself.
    embedded = None
    try:
        embedded = embed_model_metadata(save_path, overwrite=False)
    except Exception as exc:
        logger = get_logger()
        if logger:
            logger.warning(f"Metadata embedding skipped for {safe_name}: {exc}")

    manager = get_manager()
    manager._discover_models()

    # Validate that the file loads and report its contract to the client.
    spec = None
    try:
        bundle = inspect_and_load(save_path, device=manager.device)
        spec = bundle.spec_summary()
    except Exception as exc:
        logger = get_logger()
        if logger:
            logger.warning(f"Uploaded model validation failed for {safe_name}: {exc}")

    return jsonify({
        "status": "success",
        "filename": safe_name,
        "path": save_path,
        "self_describing": bool(embedded),
        "metadata": embedded,
        "spec": spec,
    })


# ---------------------------------------------------------------------------
# Developer Console model tools: inspect / embed / model-specific test
# ---------------------------------------------------------------------------

def _model_or_404(name):
    manager = get_manager()
    info = manager.models.get(name)
    if info is None:
        raise LookupError(f"Model '{name}' not found")
    return manager, info


def _effective_ood(bundle):
    """Embedded OOD > matching outputs/ood JSON > config defaults."""
    if bundle.metadata.get("ood"):
        return dict(bundle.ood_config)
    ood_path = find_ood_json(bundle.path)
    if ood_path:
        try:
            with open(ood_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return {
                "enabled": True,
                "energy_temperature": data.get("energy_temperature",
                                               data.get("temperature", 1.0)),
                "temperature": data.get("temperature",
                                        data.get("energy_temperature", 1.0)),
                "energy_threshold": data.get("energy_threshold"),
                "msp_threshold": data.get("msp_threshold"),
                "binary_confidence_floor": data.get("binary_confidence_floor"),
                "calibrated_at": data.get("calibrated_at"),
                "percentile": data.get("percentile"),
                "calibrated_on_samples": data.get("calibrated_on_samples"),
                "source_json": ood_path,
                "source_file": f"legacy-json:{os.path.basename(ood_path)}",
            }
        except Exception:
            pass
    return dict(bundle.ood_config)


@models_bp.route('/api/models/inspect', methods=['POST'])
def inspect_selected_model():
    """Return the full self-describing contract for one model file."""
    payload = request.get_json(silent=True) or {}
    name = payload.get("model")
    try:
        manager, info = _model_or_404(name)
        bundle = manager._load_bundle(name)
        spec = get_model_spec(bundle.model_or_session, name, info)
        return jsonify({
            "status": "ok",
            "model": name,
            "path": bundle.path,
            "engine": bundle.engine,
            "self_describing": bool(bundle.metadata),
            "metadata": bundle.metadata,
            "preprocessing": bundle.spec_summary().get("preprocessing"),
            "ood": _effective_ood(bundle),
            "class_maps": {
                "cattle": bundle.cattle_map,
                "buffalo": bundle.buffalo_map,
            },
            "spec": spec,
        })
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger = get_logger()
        if logger:
            logger.error(f"Model inspect failed for '{name}': {exc}")
        return jsonify({"error": f"Inspect failed: {exc}"}), 500


@models_bp.route('/api/models/embed', methods=['POST'])
def embed_selected_model():
    """Embed class maps / preprocessing / OOD into a model file or a copy."""
    payload = request.get_json(silent=True) or {}
    name = payload.get("model")
    embed_ood = bool(payload.get("embed_ood", True))
    make_copy = bool(payload.get("copy", True))

    try:
        manager, info = _model_or_404(name)
        source = info["path"]
        ext = os.path.splitext(source)[1].lower()
        if ext not in (".pt", ".pth", ".onnx"):
            return jsonify({"error": f"Unsupported model format: {ext}"}), 400

        # Use the already-loaded bundle when present so ONNX/baked-normalization
        # detection and class maps stay identical to what inference uses.
        bundle = manager.bundles.get(name)
        if bundle is None:
            bundle = inspect_and_load(source, device=manager.device)

        ood_payload = _effective_ood(bundle) if embed_ood else None
        if ext == ".onnx":
            meta = build_model_metadata(
                spec=dict(bundle.spec),
                image_size=bundle.image_size,
                normalization_baked=bundle.normalization_baked,
                input_dtype=bundle.input_dtype,
                cattle_classes=bundle.cattle_map,
                buffalo_classes=bundle.buffalo_map,
                ood=ood_payload,
                source_checkpoint=source,
                extra={"embedded_by": "dev-console"},
            )
        else:
            ckpt = torch.load(source, map_location="cpu", weights_only=False)
            state = extract_state_dict(ckpt)
            if not state:
                return jsonify({"error": "Checkpoint has no state_dict"}), 400
            meta = build_model_metadata(
                state_dict=state,
                split_dir=SPLIT_DIR,
                ood=ood_payload,
                source_checkpoint=source,
                extra={"embedded_by": "dev-console"},
            )

        if not embed_ood:
            meta.pop("ood", None)
            meta["ood_embedded"] = False
        else:
            meta["ood_embedded"] = True

        if make_copy:
            dest_dir = os.path.join(EXPORT_DIR, "embedded")
            os.makedirs(dest_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(source))[0]
            suffix = "__embedded" if embed_ood else "__no_ood"
            dest = unique_path(os.path.join(dest_dir, f"{stem}{suffix}{ext}"))
            shutil.copy2(source, dest)
        else:
            # Replacing an ONNX file while an ORT session has it mapped can fail
            # on Windows; drop the cache first.
            manager.bundles.pop(name, None)
            info["model"] = None
            dest = source

        stored = embed_model_metadata(dest, meta=meta, overwrite=True)
        manager._discover_models()

        new_name = next(
            (candidate for candidate, data in manager.models.items()
             if os.path.abspath(data["path"]) == os.path.abspath(dest)),
            None,
        )
        logger = get_logger()
        if logger:
            logger.info(
                f"Model metadata embedded: {os.path.basename(dest)} "
                f"(ood={embed_ood}, copy={make_copy}, model={new_name or dest})")
        return jsonify({
            "status": "ok",
            "copied": make_copy,
            "path": dest,
            "model": new_name,
            "ood_embedded": embed_ood,
            "metadata": stored,
        })
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger = get_logger()
        if logger:
            logger.error(f"Model embed failed for '{name}': {exc}")
        return jsonify({"error": f"Embed failed: {exc}"}), 500


@models_bp.route('/api/models/test', methods=['POST'])
def test_selected_model():
    """Run only the selected model on one image (no advisory/CLIP overhead)."""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    model_name = request.form.get("model", "")
    try:
        repeats = int(float(request.form.get("repeats", 1)))
    except (TypeError, ValueError):
        repeats = 1

    try:
        result = run_model_test(
            get_manager(), model_name,
            request.files['image'].read(), repeats=repeats)
        return jsonify(result)
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger = get_logger()
        if logger:
            logger.error(f"Model test failed for '{model_name}': {exc}")
        return jsonify({"error": f"Model test failed: {exc}"}), 500
