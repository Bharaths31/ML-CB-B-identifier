import os
import glob
import json
import time
import torch
import torch.nn.functional as F

from src.config import PROJECT_ROOT, PORTABLE_EXPORT_DIR, SPLIT_DIR, IMAGE_SIZE
from src.model import BreedClassifier
from src.ood_detector import OODDetector
from src.config import OOD_ENABLED
from server.utils.logger import get_logger

class ModelManager:
    """Discovers, loads, and caches models for prediction."""

    def __init__(self):
        self.models = {}          # name -> {"path": ..., "model": ..., "backbone": ...}
        self.class_maps = {}      # "cattle" / "buffalo" -> {breed: idx}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ood_detector = OODDetector() if OOD_ENABLED else None
        self._load_class_maps()
        self._discover_models()

    def _load_class_maps(self):
        """Load breed label maps, prioritizing portable exports to avoid mismatch."""
        logger = get_logger()
        # 1. Try portable export directories first
        for portable_dir in glob.glob(os.path.join(PORTABLE_EXPORT_DIR, "*")):
            for species in ("cattle", "buffalo"):
                path = os.path.join(portable_dir, f"{species}_classes.json")
                if os.path.exists(path):
                    with open(path) as f:
                        self.class_maps[species] = json.load(f)
            if len(self.class_maps) == 2:
                break
                
        # 2. Fallback to data/splits/ if no portable exports found
        if not self.class_maps:
            for species in ("cattle", "buffalo"):
                path = os.path.join(SPLIT_DIR, f"{species}_classes.json")
                if os.path.exists(path):
                    with open(path) as f:
                        self.class_maps[species] = json.load(f)

        if logger:
            for sp, cm in self.class_maps.items():
                logger.info(f"Loaded {sp} class map ({len(cm)} breeds)")

    def _discover_models(self):
        """Find all available checkpoints."""
        found = {}
        logger = get_logger()
        out_dir = os.path.join(PROJECT_ROOT, "outputs")
        for root, dirs, files in os.walk(out_dir):
            if "logs" in root or "metrics" in root: continue
            for f in sorted(files):
                if f.endswith(".pt") or f.endswith(".onnx"):
                    path = os.path.join(root, f)
                    
                    if f == "model.pt" and "portable" in root:
                        d = os.path.basename(root)
                        name = f"portable/{d}"
                        backbone = "lite2" if "lite2" in d else ("lite4" if "lite4" in d else "lite2")
                        mtype = "pt"
                    else:
                        name = f.replace(".pt", "").replace(".onnx", " (ONNX)")
                        rel_dir = os.path.relpath(root, out_dir)
                        if rel_dir != "." and not rel_dir.startswith("checkpoints") and not rel_dir.startswith("export"):
                            parts = rel_dir.split(os.sep)
                            if parts[0].startswith("v") or parts[0].startswith("202"):
                                name = f"[{parts[0]}] {name}"
                                
                        backbone = "lite2" if "lite2" in f else ("lite4" if "lite4" in f else "lite2")
                        mtype = "onnx" if f.endswith(".onnx") else "pt"
                        
                    found[name] = {"path": path, "backbone": backbone, "model": None, "type": mtype}
                    
        self.models = found
        if logger:
            names = ", ".join(found.keys()) if found else "(none)"
            logger.info(f"Found {len(found)} models: {names}")

    def list_models(self):
        """Return list of available model names with metadata."""
        result = []
        for name, info in self.models.items():
            size_mb = os.path.getsize(info["path"]) / (1024 * 1024)
            result.append({
                "name": name,
                "path": info["path"],
                "backbone": info["backbone"],
                "size_mb": round(size_mb, 1),
            })
        return result

    def _human_number(self, n):
        """Format large numbers: 6234567 → '6.23M'."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def _load_model(self, name, socketio=None):
        """Load a model checkpoint into memory."""
        info = self.models[name]
        logger = get_logger()
        if info["model"] is not None:
            if logger:
                logger.debug(f"Model '{name}' already cached in memory")
            return info["model"]

        load_start = time.time()
        if logger:
            logger.info(f"Loading model '{name}' from {info['path']}")

        if info.get("type") == "onnx":
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                session = ort.InferenceSession(info["path"], providers=providers)
                info["model"] = session
                elapsed = time.time() - load_start
                if logger:
                    logger.info(f"ONNX model loaded in {elapsed:.3f}s")
                if socketio:
                    socketio.emit("model_loaded", {"model_name": name, "load_time_ms": elapsed * 1000}, namespace="/ws")
                return session
            except ImportError:
                raise RuntimeError("onnxruntime is not installed. Please `pip install onnxruntime` to test ONNX models.")

        backbone = info["backbone"]
        ckpt = torch.load(info["path"], map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        # Detect cosine/ArcFace heads
        _probe = BreedClassifier(backbone=backbone)
        _fi = len(_probe.cattle_head) - 1
        del _probe
        cosine = any(f"{h}.{_fi}.weight" in state and f"{h}.{_fi}.bias" not in state
                     for h in ("cattle_head", "buffalo_head"))
        
        binary_dim = state["binary_head.0.weight"].shape[0] if "binary_head.0.weight" in state else 512
        
        model = BreedClassifier(backbone=backbone, cosine_head=cosine, binary_dim=binary_dim)

        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state.items() if k in model_keys}
        model.load_state_dict(filtered, strict=False)
        model.to(self.device)
        model.eval()
        info["model"] = model

        elapsed = time.time() - load_start
        n_params = sum(p.numel() for p in model.parameters())
        if logger:
            logger.info(f"Model loaded in {elapsed:.3f}s ({self._human_number(n_params)} params, backbone={backbone})")
            
        if socketio:
            socketio.emit("model_loaded", {
                "model_name": name, 
                "load_time_ms": round(elapsed * 1000, 1),
                "params": self._human_number(n_params)
            }, namespace="/ws")

        return model

# Global singleton
manager = None

def init_manager():
    global manager
    if manager is None:
        manager = ModelManager()
    return manager

def get_manager():
    return manager
