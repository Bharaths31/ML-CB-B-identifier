import os
import json
import threading
from typing import Any, Dict, Optional
from src.config import PROJECT_ROOT

SHARED_CONFIG_PATH = os.path.join(PROJECT_ROOT, "outputs", "app_config.json")

DEFAULT_APP_CONFIG = {
    "version": 1,
    "active_model": None,  # string name or None for default
    "ood": {
        "enabled": True,
        "selected_profile": "auto",  # filename in outputs/ood/ or "auto"
        "energy_threshold": -5.0,
        "msp_threshold": 0.5,
        "temperature": 1.0,
    },
    "display": {
        "confidence_mode": "percentage",  # "percentage", "scaled", "clamped", "badge_only", "hide"
        "confidence_floor_pct": 80.0,
        "top_k": 3,
        "min_breeds": 1,
        "max_breeds": 5,
        "min_breed_confidence_pct": 5.0,
        "show_species_confidence": True,
        "show_species_badge": True,
        "show_escalation_warning": True,
        "show_technical_details": False,
        "theme": "dark",  # "dark" or "light"
    },
    "dataset": {
        "min_images_per_breed": 15,
        "max_images_per_breed": 500,
    },
    "inference": {
        "device_override": None,  # None, "cuda", "cpu"
        "batch_size": 16,
    }
}


def _deep_merge(source: dict, destination: dict) -> dict:
    """Recursively merge source into destination."""
    for key, value in source.items():
        if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
            destination[key] = _deep_merge(value, destination[key].copy())
        else:
            destination[key] = value
    return destination


class SharedConfigDB:
    """
    Thread-safe, file-backed shared JSON database for application settings.
    Provides dot-notation access, atomic writes, and WebSocket change notifications.
    """

    def __init__(self, filepath: str = SHARED_CONFIG_PATH, socketio=None):
        self.filepath = filepath
        self.socketio = socketio
        self.lock = threading.Lock()
        self._config = {}
        self.load()

    def set_socketio(self, socketio):
        """Bind or update the socketio instance for event emissions."""
        self.socketio = socketio

    def load(self) -> Dict[str, Any]:
        """Load configuration from JSON file, creating defaults if missing or corrupted."""
        with self.lock:
            if not os.path.exists(self.filepath):
                self._config = json.loads(json.dumps(DEFAULT_APP_CONFIG))
                self._save_unlocked()
                return self._config

            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Merge loaded data over defaults so missing new keys get filled
                    merged = json.loads(json.dumps(DEFAULT_APP_CONFIG))
                    self._config = _deep_merge(data, merged)
            except Exception:
                self._config = json.loads(json.dumps(DEFAULT_APP_CONFIG))
                self._save_unlocked()

            return self._config

    def _save_unlocked(self):
        """Internal save without acquiring lock (caller must hold self.lock)."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        temp_file = self.filepath + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2)
        os.replace(temp_file, self.filepath)

    def save(self):
        """Save current in-memory config to disk."""
        with self.lock:
            self._save_unlocked()

    def get_all(self) -> Dict[str, Any]:
        """Return a copy of the entire config tree."""
        with self.lock:
            return json.loads(json.dumps(self._config))

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a value by dot-notation key path, e.g. 'ood.enabled'."""
        with self.lock:
            keys = dotted_key.split(".")
            curr = self._config
            for k in keys:
                if isinstance(curr, dict) and k in curr:
                    curr = curr[k]
                else:
                    return default
            return json.loads(json.dumps(curr)) if isinstance(curr, (dict, list)) else curr

    def set(self, dotted_key: str, value: Any, notify: bool = True):
        """Set a value by dot-notation key path and optionally emit a change event."""
        with self.lock:
            keys = dotted_key.split(".")
            curr = self._config
            for k in keys[:-1]:
                if k not in curr or not isinstance(curr[k], dict):
                    curr[k] = {}
                curr = curr[k]
            curr[keys[-1]] = value
            # Cleanup any root-level dotted key if present
            if "." in dotted_key and dotted_key in self._config:
                del self._config[dotted_key]
            # Synchronize max_breeds and top_k
            if dotted_key == "display.max_breeds":
                self._config.setdefault("display", {})["top_k"] = value
            elif dotted_key == "display.top_k":
                self._config.setdefault("display", {})["max_breeds"] = value
            self._save_unlocked()
            snapshot = json.loads(json.dumps(self._config))

        if notify and self.socketio:
            try:
                self.socketio.emit(
                    "config_changed",
                    {"key": dotted_key, "value": value, "config": snapshot},
                    namespace="/ws"
                )
            except Exception:
                pass

    def update(self, updates: Dict[str, Any], notify: bool = True):
        """Update multiple keys from a dictionary and save atomically."""
        with self.lock:
            for k, v in updates.items():
                if "." in k:
                    keys = k.split(".")
                    curr = self._config
                    for part in keys[:-1]:
                        if part not in curr or not isinstance(curr[part], dict):
                            curr[part] = {}
                        curr = curr[part]
                    curr[keys[-1]] = v
                    if k in self._config:
                        del self._config[k]
                else:
                    if isinstance(v, dict) and k in self._config and isinstance(self._config[k], dict):
                        self._config[k] = _deep_merge(v, self._config[k].copy())
                    else:
                        self._config[k] = v

            # Synchronize max_breeds and top_k
            disp = self._config.get("display", {})
            if isinstance(disp, dict):
                if "max_breeds" in updates or "display.max_breeds" in updates:
                    disp["top_k"] = disp["max_breeds"]
                elif "top_k" in updates or "display.top_k" in updates:
                    disp["max_breeds"] = disp["top_k"]

            self._save_unlocked()
            snapshot = json.loads(json.dumps(self._config))

        if notify and self.socketio:
            try:
                self.socketio.emit(
                    "config_changed",
                    {"updates": updates, "config": snapshot},
                    namespace="/ws"
                )
            except Exception:
                pass

    def reset_to_defaults(self, notify: bool = True) -> Dict[str, Any]:
        """Reset config to default values and save."""
        with self.lock:
            self._config = json.loads(json.dumps(DEFAULT_APP_CONFIG))
            self._save_unlocked()
            snapshot = json.loads(json.dumps(self._config))

        if notify and self.socketio:
            try:
                self.socketio.emit(
                    "config_changed",
                    {"reset": True, "config": snapshot},
                    namespace="/ws"
                )
            except Exception:
                pass
        return snapshot


# Module singleton
_instance: Optional[SharedConfigDB] = None

def get_shared_config(socketio=None) -> SharedConfigDB:
    global _instance
    if _instance is None:
        _instance = SharedConfigDB(socketio=socketio)
    elif socketio is not None and _instance.socketio is None:
        _instance.set_socketio(socketio)
    return _instance
