import os
import glob
import json
import time
import torch
import torch.nn.functional as F

from src.config import PROJECT_ROOT, PORTABLE_EXPORT_DIR, SPLIT_DIR, IMAGE_SIZE, OOD_DIR
from src.model import BreedClassifier, OODDetector
from src.config import OOD_ENABLED
from server.utils.logger import get_logger
from server.shared_config import get_shared_config

# Disable NNPACK warnings on CPU
if hasattr(torch.backends, 'nnpack'):
    try:
        torch.backends.nnpack.set_flags(False)
    except Exception:
        pass

class ModelManager:
    """Discovers, loads, and caches models for prediction."""

    def __init__(self):
        self.models = {}          # name -> {"path": ..., "model": ..., "backbone": ...}
        self.class_maps = {}      # "cattle" / "buffalo" -> {breed: idx}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        db = get_shared_config()
        ood_enabled = db.get("ood.enabled", OOD_ENABLED)
        selected_prof = db.get("ood.selected_profile", "auto")
        prof_path = None
        if selected_prof != "auto":
            candidates = [
                os.path.join(OOD_DIR, selected_prof),
                os.path.join(PROJECT_ROOT, "outputs", selected_prof)
            ]
            for c in candidates:
                if os.path.exists(c):
                    prof_path = c
                    break
        self.ood_detector = OODDetector(config_path=prof_path) if ood_enabled else None
        self._load_class_maps()
        self._discover_models()

    def get_active_model_name(self):
        """Return currently configured active model name or sensible default."""
        db = get_shared_config()
        active = db.get("active_model")
        if active and active in self.models:
            return active
        for name in self.models:
            if "portable" in name:
                return name
        if self.models:
            return next(iter(self.models.keys()))
        return None

    def set_active_model(self, name, socketio=None):
        """Set the active model across the entire application."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")
        db = get_shared_config(socketio)
        db.set("active_model", name, notify=True)
        if socketio:
            try:
                socketio.emit("model_switched", {"active_model": name}, namespace="/ws")
            except Exception:
                pass
        return name

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

    def recalibrate_ood(self, model_name, data_dir=None, split="val",
                        percentile=None, max_batches=None, max_images=None,
                        temperature=None, binary_floor=None):
        """Recalibrate OOD thresholds for the selected model.

        With ``data_dir`` the folder is walked for breed-named images; without
        it the validation/test split is used. Returns the saved threshold dict.
        """
        from src.config import (
            OOD_BINARY_CONFIDENCE_FLOOR,
            OOD_ENERGY_TEMPERATURE,
            OOD_PERCENTILE,
        )

        model = self._load_model(model_name)
        if not isinstance(model, torch.nn.Module):
            return {"error": "OOD calibration needs a PyTorch .pt checkpoint (not ONNX)"}

        percentile = OOD_PERCENTILE if percentile is None else float(percentile)
        temperature = OOD_ENERGY_TEMPERATURE if temperature is None else float(temperature)
        binary_floor = (
            OOD_BINARY_CONFIDENCE_FLOOR if binary_floor is None else float(binary_floor)
        )

        if data_dir:
            from src.ood_calibration import calibrate_from_folder, resolve_data_dir

            thresholds = calibrate_from_folder(
                model,
                resolve_data_dir(data_dir),
                self.class_maps,
                self.device,
                percentile=percentile,
                max_images=max_images,
                temperature=temperature,
                binary_floor=binary_floor,
            )
        else:
            from src.data_pipeline import get_dataloaders

            loaders = get_dataloaders(batch_size=32, num_workers=0)
            if loaders is None:
                return {"error": "no data/splits found for calibration"}
            _, val_loader, test_loader = loaders
            loader = val_loader if split == "val" else test_loader
            thresholds = OODDetector.calibrate(
                model,
                loader,
                self.device,
                percentile=percentile,
                temperature=temperature,
                binary_floor=binary_floor,
            )
            # The split path uses the full split; max_batches is accepted for
            # API compatibility but not needed for manual recalibration.
            thresholds["n_samples"] = thresholds.get("calibrated_on_samples")
            thresholds["energy_mean"] = thresholds.get("energy_stats", {}).get("mean")
            thresholds["msp_mean"] = thresholds.get("msp_stats", {}).get("mean")

        thresholds["model_name"] = model_name
        path = OODDetector.save(thresholds, model_name)
        thresholds["path"] = path

        db = get_shared_config()
        if db.get("ood.enabled", OOD_ENABLED):
            self.ood_detector = OODDetector(model_name=model_name)
        else:
            self.ood_detector = None
        return thresholds

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
        try:
            ckpt = torch.load(info["path"], map_location="cpu", weights_only=False)
        except Exception:
            try:
                ckpt = torch.jit.load(info["path"], map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint '{info['path']}': {e}")

        # If the loaded object is already a TorchScript or PyTorch Module
        if isinstance(ckpt, torch.nn.Module):
            model = ckpt
            model.to(self.device)
            model.eval()
            info["model"] = model
            elapsed = time.time() - load_start
            n_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
            if logger:
                logger.info(f"Model loaded in {elapsed:.3f}s ({self._human_number(n_params)} params, backbone={backbone})")
            if socketio:
                socketio.emit("model_loaded", {
                    "model_name": name,
                    "load_time_ms": round(elapsed * 1000, 1),
                    "params": self._human_number(n_params)
                }, namespace="/ws")
            return model

        # Handle different checkpoint formats
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format in {info['path']}: {type(ckpt)}")

        # Detect architecture dimensions dynamically from checkpoint weights
        binary_dim = state["binary_head.0.weight"].shape[0] if "binary_head.0.weight" in state else 512
        breed_dim = state["cattle_head.0.weight"].shape[0] if "cattle_head.0.weight" in state else 384

        cattle_weights = [k for k in state if k.startswith("cattle_head.") and k.endswith(".weight")]
        buffalo_weights = [k for k in state if k.startswith("buffalo_head.") and k.endswith(".weight")]

        def _layer_num(k):
            try:
                return int(k.split(".")[1])
            except (IndexError, ValueError):
                return -1

        cattle_weights.sort(key=_layer_num)
        buffalo_weights.sort(key=_layer_num)

        last_cattle = cattle_weights[-1] if cattle_weights else "cattle_head.8.weight"
        last_buffalo = buffalo_weights[-1] if buffalo_weights else "buffalo_head.8.weight"

        num_cattle = state[last_cattle].shape[0] if last_cattle in state else 46
        num_buffalo = state[last_buffalo].shape[0] if last_buffalo in state else 11

        # Detect cosine/ArcFace heads (final layer has weight but no bias)
        cosine = False
        if last_cattle in state:
            bias_key = last_cattle[:-7] + ".bias"
            if bias_key not in state:
                cosine = True

        model = BreedClassifier(
            backbone=backbone,
            cosine_head=cosine,
            binary_dim=binary_dim,
            breed_dim=breed_dim,
            num_cattle=num_cattle,
            num_buffalo=num_buffalo,
        )

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
