import os
import torch
import torch.nn.functional as F

from .config import OOD_ENERGY_TEMPERATURE, OOD_ENERGY_THRESHOLD, OOD_MSP_THRESHOLD


class OODDetector:
    """Post-hoc energy-based out-of-distribution detector.
    
    Computes an OOD score from raw model logits. No retraining needed.
    Works with both PyTorch and ONNX models (when using numpy arrays).
    
    If ``outputs/ood_thresholds.json`` exists (produced by
    ``scripts/calibrate_ood.py``), thresholds are auto-loaded from it so
    you don't need to manually update ``src/config.py``.
    """
    
    CALIBRATION_FILE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "ood_thresholds.json"
    )
    
    def __init__(self, temperature=OOD_ENERGY_TEMPERATURE,
                 energy_threshold=OOD_ENERGY_THRESHOLD,
                 msp_threshold=OOD_MSP_THRESHOLD):
        # Auto-load calibrated thresholds if available
        if os.path.exists(self.CALIBRATION_FILE):
            try:
                import json
                with open(self.CALIBRATION_FILE) as f:
                    cal = json.load(f)
                temperature = cal.get("energy_temperature", temperature)
                energy_threshold = cal.get("energy_threshold", energy_threshold)
                msp_threshold = cal.get("msp_threshold", msp_threshold)
            except Exception:
                pass  # fall back to defaults
        self.T = temperature
        self.energy_threshold = energy_threshold
        self.msp_threshold = msp_threshold

    def compute_energy(self, *logits_list):
        """E(x) = -T * log(Σ exp(f_i / T)) across all logit heads."""
        is_numpy = not isinstance(logits_list[0], torch.Tensor)
        
        if is_numpy:
            import numpy as np
            # numpy implementation
            combined_logits = np.concatenate(logits_list, axis=-1)
            # stable logsumexp
            max_val = np.max(combined_logits / self.T, axis=-1, keepdims=True)
            sum_exp = np.sum(np.exp(combined_logits / self.T - max_val), axis=-1)
            energy = -self.T * (np.squeeze(max_val, axis=-1) + np.log(sum_exp))
            return energy
            
        else:
            # torch implementation
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
        
        A sample is considered OOD if its energy is strictly greater than the
        energy threshold, OR if its binary maximum softmax probability (MSP) is
        lower than the MSP threshold. High energy = OOD.
        """
        energy = self.compute_energy(binary_logits, cattle_logits, buffalo_logits)
        msp = self.compute_msp(binary_logits)
        
        # handle batch or single sample
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
        is_ood_flag = ood_by_energy or ood_by_msp
        
        # Using energy_val as the primary OOD score (higher is more anomalous)
        return is_ood_flag, energy_val, {
            "energy": energy_val,
            "energy_threshold": self.energy_threshold,
            "msp": msp_val,
            "msp_threshold": self.msp_threshold,
            "ood_by_energy": ood_by_energy,
            "ood_by_msp": ood_by_msp
        }
