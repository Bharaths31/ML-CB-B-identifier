import os
import glob
import json
import time
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename

from src.config import PROJECT_ROOT, OOD_DIR
from src.ood_detector import OODDetector
from server.model_manager import get_manager
from server.shared_config import get_shared_config
from server.utils.logger import get_logger

ood_bp = Blueprint('ood', __name__)

@ood_bp.route('/api/ood/profiles', methods=['GET'])
def list_ood_profiles():
    """List all available OOD calibration profiles with metadata."""
    profiles = []
    logger = get_logger()

    # Ensure ood dir exists
    os.makedirs(OOD_DIR, exist_ok=True)

    # 1. Global calibration file if exists
    global_file = os.path.join(PROJECT_ROOT, "outputs", "ood_thresholds.json")
    if os.path.exists(global_file):
        try:
            with open(global_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            stat = os.stat(global_file)
            profiles.append({
                "filename": "ood_thresholds.json",
                "path": global_file,
                "is_global": True,
                "model_name": data.get("model_name", "Global Fallback"),
                "energy_threshold": data.get("energy_threshold"),
                "msp_threshold": data.get("msp_threshold"),
                "temperature": data.get("energy_temperature", 1.0),
                "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            })
        except Exception as e:
            if logger:
                logger.error(f"Error reading {global_file}: {e}")

    # 2. Profiles in outputs/ood/
    for fpath in sorted(glob.glob(os.path.join(OOD_DIR, "*.json"))):
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            stat = os.stat(fpath)
            profiles.append({
                "filename": fname,
                "path": fpath,
                "is_global": False,
                "model_name": data.get("model_name", fname.replace(".json", "")),
                "energy_threshold": data.get("energy_threshold"),
                "msp_threshold": data.get("msp_threshold"),
                "temperature": data.get("energy_temperature", 1.0),
                "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            })
        except Exception as e:
            if logger:
                logger.error(f"Error reading {fpath}: {e}")

    return jsonify({"profiles": profiles})


@ood_bp.route('/api/ood/active', methods=['GET'])
def get_active_ood():
    """Get active OOD configuration and currently resolved detector status."""
    socketio = current_app.extensions.get('socketio')
    db = get_shared_config(socketio)
    manager = get_manager()

    ood_cfg = db.get("ood", {})
    detector_status = {}

    if manager and manager.ood_detector:
        det = manager.ood_detector
        source_val = getattr(det, "source_file", getattr(det, "source", None))
        prof_name = getattr(det, "profile_name", (os.path.basename(source_val) if source_val else "default"))
        detector_status = {
            "initialized": True,
            "profile_name": prof_name,
            "source_file": source_val,
            "energy_threshold": getattr(det, "energy_threshold", -5.0),
            "msp_threshold": getattr(det, "msp_threshold", 0.5),
            "temperature": getattr(det, "T", 1.0),
        }
    else:
        detector_status = {
            "initialized": False,
            "profile_name": None,
            "source_file": None,
        }

    return jsonify({
        "config": ood_cfg,
        "detector": detector_status
    })


@ood_bp.route('/api/ood/select', methods=['POST'])
def select_profile():
    """Select active OOD profile ('auto' or specific filename)."""
    socketio = current_app.extensions.get('socketio')
    db = get_shared_config(socketio)
    manager = get_manager()
    logger = get_logger()

    data = request.get_json() or {}
    profile = data.get("profile")
    if not profile:
        return jsonify({"error": "Missing 'profile' field"}), 400

    profile_path = None
    if profile != "auto":
        candidates = [
            os.path.join(OOD_DIR, profile),
            os.path.join(PROJECT_ROOT, "outputs", profile)
        ]
        for c in candidates:
            if os.path.exists(c):
                profile_path = c
                break
        if not profile_path:
            return jsonify({"error": f"Profile '{profile}' not found"}), 404

    db.set("ood.selected_profile", profile, notify=True)

    # Re-initialize OOD detector on manager if enabled
    if manager and db.get("ood.enabled", True):
        manager.ood_detector = OODDetector(config_path=profile_path)
        if profile_path and manager.ood_detector.metadata:
            cal = manager.ood_detector.metadata
            db.set("ood.energy_threshold", cal.get("energy_threshold", db.get("ood.energy_threshold")))
            db.set("ood.msp_threshold", cal.get("msp_threshold", db.get("ood.msp_threshold")))
            db.set("ood.temperature", cal.get("energy_temperature", db.get("ood.temperature")))

    if logger:
        logger.info(f"OOD profile selected: {profile}")

    if socketio:
        socketio.emit("ood_status_changed", {
            "profile": profile,
            "enabled": db.get("ood.enabled", True),
            "profile_path": profile_path
        }, namespace="/ws")

    return jsonify({"status": "ok", "profile": profile, "profile_path": profile_path})


@ood_bp.route('/api/ood/toggle', methods=['POST'])
def toggle_ood():
    """Toggle OOD detection on or off."""
    socketio = current_app.extensions.get('socketio')
    db = get_shared_config(socketio)
    manager = get_manager()
    logger = get_logger()

    data = request.get_json() or {}
    if "enabled" not in data:
        return jsonify({"error": "Missing 'enabled' boolean"}), 400

    enabled = bool(data["enabled"])
    db.set("ood.enabled", enabled, notify=True)

    if manager:
        if enabled:
            selected_profile = db.get("ood.selected_profile", "auto")
            profile_path = None
            if selected_profile != "auto":
                candidate = os.path.join(OOD_DIR, selected_profile)
                if os.path.exists(candidate):
                    profile_path = candidate
            manager.ood_detector = OODDetector(config_path=profile_path)
        else:
            manager.ood_detector = None

    if logger:
        logger.info(f"OOD enabled set to: {enabled}")

    if socketio:
        socketio.emit("ood_status_changed", {"enabled": enabled}, namespace="/ws")

    return jsonify({"status": "ok", "enabled": enabled})


@ood_bp.route('/api/ood/thresholds', methods=['POST'])
def update_thresholds():
    """Update custom OOD thresholds."""
    socketio = current_app.extensions.get('socketio')
    db = get_shared_config(socketio)
    manager = get_manager()

    data = request.get_json() or {}
    updates = {}
    if "energy_threshold" in data:
        updates["ood.energy_threshold"] = float(data["energy_threshold"])
    if "msp_threshold" in data:
        updates["ood.msp_threshold"] = float(data["msp_threshold"])
    if "temperature" in data:
        updates["ood.temperature"] = float(data["temperature"])

    for k, v in updates.items():
        db.set(k, v, notify=False)

    # Apply directly to active detector if initialized
    if manager and manager.ood_detector:
        if "energy_threshold" in data:
            manager.ood_detector.energy_threshold = float(data["energy_threshold"])
        if "msp_threshold" in data:
            manager.ood_detector.msp_threshold = float(data["msp_threshold"])
        if "temperature" in data:
            manager.ood_detector.T = float(data["temperature"])

    if socketio:
        socketio.emit("config_changed", {"updates": updates}, namespace="/ws")

    return jsonify({"status": "ok", "updates": updates})


@ood_bp.route('/api/ood/upload', methods=['POST'])
def upload_profile():
    """Upload a new OOD JSON configuration profile."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith('.json'):
        return jsonify({"error": "Only .json files are allowed"}), 400

    filename = secure_filename(file.filename)
    os.makedirs(OOD_DIR, exist_ok=True)
    save_path = os.path.join(OOD_DIR, filename)

    try:
        # Validate that it's valid JSON before saving
        content = file.read().decode('utf-8')
        data = json.loads(content)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return jsonify({
            "status": "success",
            "filename": filename,
            "model_name": data.get("model_name"),
            "energy_threshold": data.get("energy_threshold"),
            "msp_threshold": data.get("msp_threshold"),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to save profile: {str(e)}"}), 400
