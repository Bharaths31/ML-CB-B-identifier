import os
import json
from flask import Blueprint, jsonify, request
from src.config import LOGS_DIR
from server.utils.logger import get_logger

config_bp = Blueprint('config', __name__)

PRESENTER_CONFIG_PATH = os.path.join(LOGS_DIR, "presenter_config.json")

DEFAULT_PRESENTER_CONFIG = {
    "presenter_title": "🐄 Breed Classifier — Demonstration",
    "presenter_subtitle": "AI-Powered Cattle & Buffalo Breed Identification",
    "confidence_display_mode": "scaled",  # "scaled", "clamped", "badge_only", "hide", "percentage"
    "confidence_floor_pct": 82.0,
    "hide_species_confidence": True,
    "hide_species_badge": False,
    "hide_top5_list": False,
    "top5_count": 3,
    "min_breed_confidence_pct": 10.0,
    "hide_info_footer": True,
    "hide_model_selector": False,
    "hide_batch_tab": False,
    "hide_batch_summary_stats": True,
}

def load_presenter_config():
    """Load presenter config from JSON, creating defaults if missing."""
    try:
        with open(PRESENTER_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            merged = {**DEFAULT_PRESENTER_CONFIG, **cfg}
            return merged
    except (FileNotFoundError, json.JSONDecodeError):
        save_presenter_config(DEFAULT_PRESENTER_CONFIG)
        return dict(DEFAULT_PRESENTER_CONFIG)

def save_presenter_config(config):
    """Save presenter config to JSON."""
    os.makedirs(os.path.dirname(PRESENTER_CONFIG_PATH), exist_ok=True)
    with open(PRESENTER_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

@config_bp.route('/api/presenter-config', methods=['GET'])
def get_config():
    cfg = load_presenter_config()
    return jsonify(cfg)

@config_bp.route('/api/presenter-config', methods=['POST'])
def update_config():
    try:
        cfg = request.get_json()
        if not cfg:
            return jsonify({"error": "Invalid JSON"}), 400
            
        save_presenter_config(cfg)
        logger = get_logger()
        if logger:
            logger.info(f"Presenter config saved: {cfg}")
            
        return jsonify({"status": "saved"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
