import datetime
import glob
import json
import os
import re
import torch
import torch.nn.functional as F

from .config import (
    PROJECT_ROOT,
    OOD_DIR,
    OOD_ENERGY_TEMPERATURE,
    OOD_ENERGY_THRESHOLD,
    OOD_MSP_THRESHOLD,
    OOD_BINARY_CONFIDENCE_FLOOR,
    OOD_PERCENTILE,
    PORTABLE_EXPORT_DIR,
)


class OODDetector:
    """Post-hoc energy-based out-of-distribution detector.

    Computes an OOD score from raw model logits without requiring retraining.
    Works with both PyTorch and ONNX models (torch tensors and numpy arrays).

    Supports:
      1. Per-model calibrated thresholds (e.g. outputs/ood/<model_name>.json).
      2. Custom selected OOD configuration files.
      3. Global fallback to outputs/ood_thresholds.json or config defaults.
      4. Auto-calibration after model training.
    """

    DEFAULT_CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "outputs", "ood_thresholds.json")

    def __init__(self, model_name=None, config_path=None,
                 temperature=OOD_ENERGY_TEMPERATURE,
                 energy_threshold=OOD_ENERGY_THRESHOLD,
                 msp_threshold=OOD_MSP_THRESHOLD,
                 binary_confidence_floor=OOD_BINARY_CONFIDENCE_FLOOR):
        self.source_file = None
        self.profile_name = "default"
        self.metadata = {}

        # 1. Resolve configuration file if custom path or model name is given
        resolved_path = None
        if config_path and os.path.exists(config_path):
            resolved_path = config_path
        elif model_name:
            resolved_path = self.resolve_profile_path(model_name)

        if resolved_path is None and os.path.exists(self.DEFAULT_CALIBRATION_FILE):
            resolved_path = self.DEFAULT_CALIBRATION_FILE

        if resolved_path and os.path.exists(resolved_path):
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    cal = json.load(f)
                temperature = cal.get("energy_temperature", temperature)
                energy_threshold = cal.get("energy_threshold", energy_threshold)
                msp_threshold = cal.get("msp_threshold", msp_threshold)
                binary_confidence_floor = cal.get("binary_confidence_floor",
                                                   binary_confidence_floor)
                self.source_file = resolved_path
                self.profile_name = os.path.basename(resolved_path)
                self.metadata = cal
            except Exception:
                pass

        self.T = float(temperature)
        self.energy_threshold = float(energy_threshold)
        self.msp_threshold = float(msp_threshold)
        self.binary_confidence_floor = float(binary_confidence_floor)

    @classmethod
    def clean_name(cls, name: str) -> str:
        """Strip directory prefixes, file extensions, and UI tags from model name."""
        if not name:
            return "model"
        base = os.path.basename(name)
        # remove [v17] or similar prefix
        base = re.sub(r"^\[.*?\]\s*", "", base)
        # remove (ONNX) or similar suffix
        base = re.sub(r"\s*\(ONNX\)$", "", base, flags=re.IGNORECASE)
        # remove extensions
        for ext in (".pt", ".onnx", ".json"):
            if base.lower().endswith(ext):
                base = base[:-len(ext)]
        # replace illegal characters
        base = re.sub(r"[^\w\-_.]", "_", base)
        return base.strip()

    @classmethod
    def resolve_profile_path(cls, model_name: str) -> str | None:
        """Find the matching OOD calibration JSON for a given model."""
        if not model_name:
            return None

        # Direct path check
        if os.path.isabs(model_name) and os.path.exists(model_name):
            return model_name

        clean = cls.clean_name(model_name)

        # 1. Exact match in outputs/ood/<clean>.json
        candidate = os.path.join(OOD_DIR, f"{clean}.json")
        if os.path.exists(candidate):
            return candidate

        # 2. Check candidate with _ood suffix
        candidate_ood = os.path.join(OOD_DIR, f"{clean}_ood.json")
        if os.path.exists(candidate_ood):
            return candidate_ood

        # 3. Check portable export directories
        portable_dirs = glob.glob(os.path.join(PORTABLE_EXPORT_DIR, "*"))
        for pdir in portable_dirs:
            if clean in os.path.basename(pdir):
                p_ood = os.path.join(pdir, "ood_thresholds.json")
                if os.path.exists(p_ood):
                    return p_ood

        # 4. Partial substring match in outputs/ood/
        if os.path.exists(OOD_DIR):
            for fname in os.listdir(OOD_DIR):
                if fname.endswith(".json"):
                    stem = fname[:-5]
                    if stem in clean or clean in stem:
                        return os.path.join(OOD_DIR, fname)

        # 5. Global fallback
        if os.path.exists(cls.DEFAULT_CALIBRATION_FILE):
            return cls.DEFAULT_CALIBRATION_FILE

        return None

    @classmethod
    def calibrate(cls, model, loader, device, percentile=OOD_PERCENTILE,
                  temperature=OOD_ENERGY_TEMPERATURE,
                  binary_floor=OOD_BINARY_CONFIDENCE_FLOOR) -> dict:
        """Calibrate energy and MSP thresholds on an in-distribution DataLoader.

        Returns a dictionary of calibrated parameters and distribution stats.
        """
        device = torch.device(device) if isinstance(device, str) else device
        model.eval()

        detector = cls(temperature=temperature,
                       binary_confidence_floor=binary_floor)
        energies = []
        msps = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device)

                with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    out = model(images)

                b_logits = out["binary"]
                c_logits = out["cattle"]
                buf_logits = out["buffalo"]

                e_batch = detector.compute_energy(b_logits, c_logits, buf_logits)
                m_batch = detector.compute_msp(b_logits)

                energies.extend(e_batch.detach().cpu().tolist())
                msps.extend(m_batch.detach().cpu().tolist())

        if not energies:
            raise ValueError("[OODDetector] Calibration failed: loader yielded 0 samples.")

        energies_t = torch.tensor(energies, dtype=torch.float32)
        msps_t = torch.tensor(msps, dtype=torch.float32)

        energy_threshold = torch.quantile(energies_t, percentile / 100.0).item()
        msp_threshold = torch.quantile(msps_t, (100.0 - percentile) / 100.0).item()

        return {
            "energy_temperature": float(temperature),
            "energy_threshold": round(float(energy_threshold), 4),
            "msp_threshold": round(float(msp_threshold), 4),
            "binary_confidence_floor": float(binary_floor),
            "percentile": float(percentile),
            "calibrated_on_samples": len(energies),
            "calibrated_at": datetime.datetime.now().isoformat(),
            "energy_stats": {
                "mean": round(float(energies_t.mean().item()), 4),
                "std": round(float(energies_t.std().item()), 4),
                "min": round(float(energies_t.min().item()), 4),
                "max": round(float(energies_t.max().item()), 4),
            },
            "msp_stats": {
                "mean": round(float(msps_t.mean().item()), 4),
                "std": round(float(msps_t.std().item()), 4),
                "min": round(float(msps_t.min().item()), 4),
                "max": round(float(msps_t.max().item()), 4),
            }
        }

    @classmethod
    def save(cls, thresholds: dict, name: str, out_dir: str = None) -> str:
        """Save calibrated thresholds to JSON with appropriate naming."""
        clean = cls.clean_name(name)
        target_dir = out_dir if out_dir is not None else OOD_DIR
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, f"{clean}.json")
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)

        # Update global fallback outputs/ood_thresholds.json
        try:
            with open(cls.DEFAULT_CALIBRATION_FILE, "w", encoding="utf-8") as f:
                json.dump(thresholds, f, indent=2)
        except Exception:
            pass

        return target_path

    @classmethod
    def list_available_profiles(cls) -> list:
        """Enumerate all available OOD calibration files and their metrics."""
        profiles = []
        seen_paths = set()

        def _add_profile(path, label=None, is_default=False):
            if path in seen_paths or not os.path.exists(path):
                return
            seen_paths.add(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                fname = label or os.path.basename(path)
                profiles.append({
                    "id": fname,
                    "filename": fname,
                    "path": path,
                    "is_default": is_default,
                    "energy_threshold": data.get("energy_threshold", OOD_ENERGY_THRESHOLD),
                    "msp_threshold": data.get("msp_threshold", OOD_MSP_THRESHOLD),
                    "binary_confidence_floor": data.get("binary_confidence_floor", OOD_BINARY_CONFIDENCE_FLOOR),
                    "energy_temperature": data.get("energy_temperature", OOD_ENERGY_TEMPERATURE),
                    "percentile": data.get("percentile", OOD_PERCENTILE),
                    "calibrated_on_samples": data.get("calibrated_on_samples"),
                    "calibrated_at": data.get("calibrated_at"),
                })
            except Exception:
                pass

        # 1. Models in OOD_DIR
        if os.path.exists(OOD_DIR):
            for f in sorted(os.listdir(OOD_DIR)):
                if f.endswith(".json"):
                    _add_profile(os.path.join(OOD_DIR, f))

        # 2. Portable export folders
        for pdir in sorted(glob.glob(os.path.join(PORTABLE_EXPORT_DIR, "*"))):
            p_ood = os.path.join(pdir, "ood_thresholds.json")
            if os.path.exists(p_ood):
                _add_profile(p_ood, label=f"portable/{os.path.basename(pdir)}")

        # 3. Default fallback
        if os.path.exists(cls.DEFAULT_CALIBRATION_FILE):
            _add_profile(cls.DEFAULT_CALIBRATION_FILE, label="ood_thresholds.json", is_default=True)

        return profiles

    def compute_energy(self, *logits_list):
        """E(x) = -T * log(Σ exp(f_i / T)) across all logit heads."""
        is_numpy = not isinstance(logits_list[0], torch.Tensor)

        if is_numpy:
            import numpy as np
            combined_logits = np.concatenate(logits_list, axis=-1)
            max_val = np.max(combined_logits / self.T, axis=-1, keepdims=True)
            sum_exp = np.sum(np.exp(combined_logits / self.T - max_val), axis=-1)
            energy = -self.T * (np.squeeze(max_val, axis=-1) + np.log(sum_exp))
            return energy
        else:
            combined_logits = torch.cat(logits_list, dim=-1)
            energy = -self.T * torch.logsumexp(combined_logits / self.T, dim=-1)
            return energy

    def compute_msp(self, binary_logits):
        """Max softmax probability from binary head (cattle-vs-buffalo)."""
        is_numpy = not isinstance(binary_logits, torch.Tensor)
        if is_numpy:
            import numpy as np
            exp_logits = np.exp(binary_logits - np.max(binary_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return np.max(probs, axis=-1)
        else:
            probs = F.softmax(binary_logits, dim=-1)
            return torch.max(probs, dim=-1)[0]

    def is_ood(self, binary_logits, cattle_logits, buffalo_logits):
        """Returns (is_ood: bool, ood_score: float, details: dict).

        A sample is considered OOD if:
          1. Its energy is strictly greater than the energy threshold, OR
          2. Its binary maximum softmax probability (MSP) is lower than the
             MSP threshold, OR
          3. Its binary confidence (max of P(cattle), P(buffalo)) is below
             the binary confidence floor.
        """
        energy = self.compute_energy(binary_logits, cattle_logits, buffalo_logits)
        msp = self.compute_msp(binary_logits)

        is_numpy = not isinstance(energy, torch.Tensor)
        if is_numpy:
            import numpy as np
            energy_val = float(np.atleast_1d(energy)[0])
            msp_val = float(np.atleast_1d(msp)[0])
        else:
            energy_val = energy.item() if energy.numel() == 1 else energy[0].item()
            msp_val = msp.item() if msp.numel() == 1 else msp[0].item()

        ood_by_energy = energy_val > self.energy_threshold
        ood_by_msp = msp_val < self.msp_threshold
        ood_by_binary_floor = msp_val < self.binary_confidence_floor
        is_ood_flag = ood_by_energy or ood_by_msp or ood_by_binary_floor

        return is_ood_flag, energy_val, {
            "energy": energy_val,
            "energy_threshold": self.energy_threshold,
            "msp": msp_val,
            "msp_threshold": self.msp_threshold,
            "binary_confidence_floor": self.binary_confidence_floor,
            "ood_by_energy": ood_by_energy,
            "ood_by_msp": ood_by_msp,
            "ood_by_binary_floor": ood_by_binary_floor,
            "source_file": self.source_file,
            "profile_name": self.profile_name,
        }
