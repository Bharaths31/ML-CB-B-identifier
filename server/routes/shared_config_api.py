from flask import Blueprint, jsonify, request, current_app
from server.shared_config import get_shared_config
from server.utils.logger import get_logger

shared_config_bp = Blueprint('shared_config', __name__)

@shared_config_bp.route('/api/config', methods=['GET'])
def get_config():
    """Retrieve the entire current application configuration."""
    socketio = current_app.extensions.get('socketio')
    db = get_shared_config(socketio)
    return jsonify(db.get_all())


@shared_config_bp.route('/api/config', methods=['PATCH', 'POST'])
def update_config():
    """
    Update configuration values.
    Supports either:
    1. {"key": "ood.enabled", "value": false}
    2. {"ood": {"enabled": false}, "display": {"theme": "light"}}
    """
    socketio = current_app.extensions.get('socketio')
    db = get_shared_config(socketio)
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No JSON payload provided"}), 400

    logger = get_logger()

    if "key" in payload and "value" in payload:
        key = payload["key"]
        val = payload["value"]
        db.set(key, val, notify=True)
        if logger:
            logger.info(f"Config updated [{key}]: {val}")
    else:
        db.update(payload, notify=True)
        if logger:
            logger.info(f"Config updated batch: {list(payload.keys())}")

    return jsonify({"status": "ok", "config": db.get_all()})


@shared_config_bp.route('/api/config/reset', methods=['POST'])
def reset_config():
    """Reset configuration to initial default schema."""
    socketio = current_app.extensions.get('socketio')
    db = get_shared_config(socketio)
    logger = get_logger()
    cfg = db.reset_to_defaults(notify=True)
    if logger:
        logger.info("Config reset to defaults")
    return jsonify({"status": "reset", "config": cfg})


@shared_config_bp.route('/api/config/<path:key>', methods=['GET'])
def get_config_key(key):
    """Retrieve a single key or sub-tree, e.g. /api/config/ood/enabled."""
    socketio = current_app.extensions.get('socketio')
    db = get_shared_config(socketio)
    dotted_key = key.replace("/", ".")
    val = db.get(dotted_key)
    if val is None:
        return jsonify({"error": f"Key '{dotted_key}' not found"}), 404
    return jsonify({"key": dotted_key, "value": val})
