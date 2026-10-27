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

def portable_bundle_sort_key(bundle_dir):
    """Newest-first sort key for a portable export directory.

    ISO-8601 ``exported_at`` strings sort lexicographically, so they are the
    primary key; filesystem mtime is the fallback (and tie-breaker) when an
    older bundle has no ``model_info.json``.
    """
    exported_at = ""
    info_path = os.path.join(bundle_dir, "model_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path) as fh:
                exported_at = str(json.load(fh).get("exported_at") or "")
        except Exception:
            exported_at = ""
    try:
        mtime = os.path.getmtime(bundle_dir)
    except OSError:
        mtime = 0.0
    return (exported_at, mtime)


def sorted_portable_bundles():
    """Return portable export directories, newest first."""
    dirs = [d for d in glob.glob(os.path.join(PORTABLE_EXPORT_DIR, "*"))
            if os.path.isdir(d)]
    dirs.sort(key=portable_bundle_sort_key, reverse=True)
    return dirs


def read_class_maps_from(bundle_dir):
    """Read a complete (cattle, buffalo) class-map pair from a directory."""
    maps = {}
    for species in ("cattle", "buffalo"):
        path = os.path.join(bundle_dir, f"{species}_classes.json")
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    maps[species] = json.load(fh)
            except Exception:
                return {}
    return maps if len(maps) == 2 else {}


def head_out_features(module):
    """Best-effort output width of a breed head (Linear or CosineHead)."""
    try:
        last = module[-1]
    except Exception:
        last = module
    out = getattr(last, "out_features", None)
    if isinstance(out, int):
        return out
    weight = getattr(last, "weight", None)
    if weight is not None and getattr(weight, "shape", None):
        return int(weight.shape[0])
    return None


class ModelManager:
    """Discovers, loads, and caches models for prediction."""

    def __init__(self):
        self.models = {}          # name -> {"path": ..., "model": ..., "backbone": ...}
        self.class_maps = {}      # "cattle" / "buffalo" -> {breed: idx}
        self.class_map_source = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_class_maps()
        self._discover_models()
        # Bind the newest class maps to the active checkpoint immediately so
        # /api/dev-info and the model-spec endpoint agree with inference.
        active = self.get_active_model_name()
        if active:
            self.bind_class_maps_for_model(active)
        self._bind_ood_detector(active)

    def _bind_ood_detector(self, model_name):
        """(Re)create the OOD detector for the active model/config."""
        db = get_shared_config()
        ood_enabled = db.get("ood.enabled", OOD_ENABLED)
        selected_prof = db.get("ood.selected_profile", "auto")
        if not ood_enabled:
            self.ood_detector = None
            return None

        if selected_prof == "auto":
            # Per-model profile resolution was previously skipped at startup,
            # which silently bound the global fallback instead.
            self.ood_detector = OODDetector(model_name=model_name) if model_name else OODDetector()
            return self.ood_detector

        prof_path = None
        candidates = [
            os.path.join(OOD_DIR, selected_prof),
            os.path.join(PROJECT_ROOT, "outputs", selected_prof),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                prof_path = candidate
                break
        self.ood_detector = OODDetector(config_path=prof_path)
        return self.ood_detector

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

        # Bind state before persisting, so a failed load/class-map binding
        # cannot leave the shared config pointing at an unusable checkpoint.
        self.bind_class_maps_for_model(name)
        self._bind_ood_detector(name)
        db.set("active_model", name, notify=True)

        if socketio:
            try:
                socketio.emit("model_switched", {"active_model": name}, namespace="/ws")
            except Exception:
                pass
            try:
                detector = self.ood_detector
                socketio.emit("ood_status_changed", {
                    "enabled": db.get("ood.enabled", OOD_ENABLED),
                    "profile": db.get("ood.selected_profile", "auto"),
                    "model": name,
                    "profile_name": getattr(detector, "profile_name", None) if detector else None,
                    "source_file": getattr(detector, "source_file", None) if detector else None,
                    "energy_threshold": getattr(detector, "energy_threshold", None) if detector else None,
                    "msp_threshold": getattr(detector, "msp_threshold", None) if detector else None,
                }, namespace="/ws")
            except Exception:
                pass
        return name

    def _load_class_maps(self):
        """Load the newest complete class-map pair (constructor default)."""
        logger = get_logger()
        self.class_maps = {}
        self.class_map_source = None
        for portable_dir in sorted_portable_bundles():
            maps = read_class_maps_from(portable_dir)
            if maps:
                self.class_maps = maps
                self.class_map_source = portable_dir
                if logger:
                    logger.info(
                        f"Using class maps from {os.path.basename(portable_dir)} "
                        f"(cattle={len(maps['cattle'])}, buffalo={len(maps['buffalo'])})"
                    )
                break

        # Fallback to data/splits/ only when no portable bundle is usable.
        if not self.class_maps:
            for species in ("cattle", "buffalo"):
                path = os.path.join(SPLIT_DIR, f"{species}_classes.json")
                if os.path.exists(path):
                    with open(path) as f:
                        self.class_maps[species] = json.load(f)
            if self.class_maps:
                self.class_map_source = SPLIT_DIR

        if logger:
            for sp, cm in self.class_maps.items():
                logger.info(f"Loaded {sp} class map ({len(cm)} breeds)")

    def _dims_from_path(self, name):
        """Read head widths from a checkpoint/ONNX file without keeping it."""
        info = self.models.get(name)
        if not info:
            return None
        if info.get("type") == "onnx":
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(info["path"], providers=["CPUExecutionProvider"])
                return self._dims_from_onnx(session)
            except Exception:
                return None
        try:
            ckpt = torch.load(info["path"], map_location="cpu", weights_only=False)
        except Exception:
            return None
        state = None
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        if not isinstance(state, dict):
            return None
        return self._dims_from_state_dict(state)

    @staticmethod
    def _dims_from_state_dict(state):
        def _last_weight(prefix):
            keys = [k for k in state if k.startswith(prefix) and k.endswith(".weight")]

            def _layer_num(k):
                try:
                    return int(k.split(".")[1])
                except (IndexError, ValueError):
                    return -1

            keys.sort(key=_layer_num)
            return keys[-1] if keys else None

        cattle_key = _last_weight("cattle_head.")
        buffalo_key = _last_weight("buffalo_head.")
        if not cattle_key or not buffalo_key:
            return None
        return {
            "cattle": int(state[cattle_key].shape[0]),
            "buffalo": int(state[buffalo_key].shape[0]),
        }

    @staticmethod
    def _dims_from_model(model):
        if not isinstance(model, torch.nn.Module):
            return None
        cattle = head_out_features(getattr(model, "cattle_head", None))
        buffalo = head_out_features(getattr(model, "buffalo_head", None))
        if cattle is None or buffalo is None:
            return None
        return {"cattle": int(cattle), "buffalo": int(buffalo)}

    @staticmethod
    def _dims_from_onnx(session):
        dims = {}
        try:
            for output in session.get_outputs():
                shape = output.shape or []
                if output.name in ("cattle", "buffalo"):
                    width = next((int(v) for v in reversed(shape) if isinstance(v, int) and v > 0), None)
                    if width:
                        dims[output.name] = width
        except Exception:
            return None
        return dims if len(dims) == 2 else None

    def bind_class_maps_for_model(self, name, model=None):
        """Select class maps whose length matches this checkpoint's heads.

        The previous implementation took the first portable directory returned
        by ``glob``, which could pair an older 17-buffalo map with an 18-class
        checkpoint. We now prefer a count-matched bundle and fall back to
        ``data/splits`` only if no bundle matches.
        """
        logger = get_logger()
        dims = self._dims_from_model(model) if model is not None else None
        if dims is None and model is not None and not isinstance(model, torch.nn.Module):
            dims = self._dims_from_onnx(model)
        if dims is None:
            dims = self._dims_from_path(name)

        bundles = sorted_portable_bundles()
        # Prefer the bundle that produced this exact model, then any count match.
        model_bundle = None
        if name and name.startswith("portable/"):
            model_bundle = name.split("portable/", 1)[1]
        if model_bundle:
            bundles.sort(key=lambda d: os.path.basename(d) != model_bundle)

        chosen = None
        for bundle_dir in bundles:
            maps = read_class_maps_from(bundle_dir)
            if not maps:
                continue
            if dims is None or (
                len(maps["cattle"]) == dims["cattle"]
                and len(maps["buffalo"]) == dims["buffalo"]
            ):
                chosen = (maps, bundle_dir)
                break

        if chosen is None:
            split_maps = {}
            for species in ("cattle", "buffalo"):
                path = os.path.join(SPLIT_DIR, f"{species}_classes.json")
                if os.path.exists(path):
                    with open(path) as f:
                        split_maps[species] = json.load(f)
            if len(split_maps) == 2:
                chosen = (split_maps, SPLIT_DIR)
                if dims and (
                    len(split_maps["cattle"]) != dims["cattle"]
                    or len(split_maps["buffalo"]) != dims["buffalo"]
                ):
                    if logger:
                        logger.warning(
                            f"No portable class map matches model '{name}' heads "
                            f"(cattle={dims['cattle']} buffalo={dims['buffalo']}); "
                            f"using data/splits fallback "
                            f"(cattle={len(split_maps['cattle'])} "
                            f"buffalo={len(split_maps['buffalo'])})."
                        )

        if chosen:
            self.class_maps, self.class_map_source = chosen
            info = self.models.get(name)
            if info is not None:
                info["class_map_source"] = self.class_map_source
                info["num_cattle"] = len(self.class_maps.get("cattle", {}))
                info["num_buffalo"] = len(self.class_maps.get("buffalo", {}))
            if logger:
                logger.info(
                    f"Bound class maps for '{name}': "
                    f"cattle={len(self.class_maps.get('cattle', {}))} "
                    f"buffalo={len(self.class_maps.get('buffalo', {}))} "
                    f"({os.path.basename(self.class_map_source or 'none')})"
                )
        return self.class_maps

    def class_counts(self, name=None):
        """Dynamic head/class-map counts for the active or requested model."""
        name = name or self.get_active_model_name()
        info = self.models.get(name, {}) if name else {}
        dims = None
        model = info.get("model")
        if model is not None:
            dims = self._dims_from_model(model) or self._dims_from_onnx(model)
        if dims is None and name:
            # If maps have already been bound, trust them; otherwise peek.
            dims = {
                "cattle": info.get("num_cattle"),
                "buffalo": info.get("num_buffalo"),
            }
            if not dims["cattle"] or not dims["buffalo"]:
                dims = self._dims_from_path(name)
        maps = self.class_maps if (not name or name == self.get_active_model_name()) else {}
        return {
            "cattle": dims.get("cattle") if dims else len(maps.get("cattle", {})),
            "buffalo": dims.get("buffalo") if dims else len(maps.get("buffalo", {})),
            "class_map_cattle": len(maps.get("cattle", {})) if maps else None,
            "class_map_buffalo": len(maps.get("buffalo", {})) if maps else None,
            "class_map_source": self.class_map_source if not name or name == self.get_active_model_name() else info.get("class_map_source"),
        }

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
                self.bind_class_maps_for_model(name, model=session)
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
            self.bind_class_maps_for_model(name, model=model)
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
        self.bind_class_maps_for_model(name, model=model)

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
